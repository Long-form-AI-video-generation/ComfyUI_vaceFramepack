"""
VAE processing, scheduler factory, RoPE embeddings, and reference image processing
for FramePack video generation.
"""

import torch
import torch.nn.functional as F
import math

from comfy.utils import common_upscale

from ..wanvideo.modules.model import rope_params
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
from ..wanvideo.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from ..wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from ..wanvideo.utils.basic_flowmatch import FlowMatchScheduler
from ..wanvideo.utils.scheduling_flow_match_lcm import FlowMatchLCMScheduler


# Constants
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


class SchedulerFactory:
    """Factory for creating schedulers"""
    
    @staticmethod
    def create_scheduler(scheduler_name, steps, shift, device, sigmas=None):
        """Setup the appropriate scheduler"""
        
        if scheduler_name == "dpm++":
            scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type="dpmsolver++")
            if sigmas is None:
                scheduler.set_timesteps(steps, device=device)
            else:
                scheduler.sigmas = sigmas.to(device)
                scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
                scheduler.num_inference_steps = len(scheduler.timesteps)
                
        elif scheduler_name == "unipc":
            scheduler = FlowUniPCMultistepScheduler(shift=shift)
            if sigmas is None:
                scheduler.set_timesteps(steps, device=device, shift=shift)
            else:
                scheduler.sigmas = sigmas.to(device)
                scheduler.timesteps = (scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
                scheduler.num_inference_steps = len(scheduler.timesteps)
                
        elif scheduler_name == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
            scheduler.set_timesteps(steps, device=device, sigmas=sigmas.tolist() if sigmas else None)
            
        elif scheduler_name == "deis":
            scheduler = DEISMultistepScheduler(
                use_flow_sigmas=True,
                prediction_type="flow_prediction",
                flow_shift=shift
            )
            scheduler.set_timesteps(steps, device=device)
            scheduler.sigmas[-1] = 1e-6
            
        elif scheduler_name == "lcm":
            scheduler = FlowMatchLCMScheduler(shift=shift)
            scheduler.set_timesteps(steps, device=device, sigmas=sigmas.tolist() if sigmas else None)
            
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
        
        return scheduler


class RoPEEmbeddings:
    """Handles RoPE embeddings setup"""
    
    @staticmethod
    def setup_rope_embeddings(model_wrapper, latent_video_length):
        """Setup RoPE embeddings for the model"""
        
        model_wrapper.rope_embedder.k = None
        model_wrapper.rope_embedder.num_frames = None
        
        d = model_wrapper.dim // model_wrapper.num_heads
        riflex_freq_index = 0
        
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)
        
        return freqs


class VAEProcessor:
    """Handles VAE encoding and decoding operations"""
    
    def __init__(self, vae, device):
        self.vae = vae
        self.device = device
    
    def encode_frames(self, frames, ref_images, masks=None, tiled_vae=False):
        """Encode frames to latent space"""
        print(f"VAE Encoding debug:")
        print(f"  Input frames type: {type(frames)}")
        if isinstance(frames, list):
             print(f"  Input frames list len: {len(frames)}")
             if len(frames) > 0:
                 print(f"  Frame 0 shape: {frames[0].shape}")
        elif isinstance(frames, torch.Tensor):
             print(f"  Input frames shape: {frames.shape}")
             
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = self.vae.encode(frames, device=self.device, tiled=tiled_vae)
        else:
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = self.vae.encode(inactive, device=self.device, tiled=tiled_vae)
            reactive = self.vae.encode(reactive, device=self.device, tiled=tiled_vae)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]
        
        self.vae.model.clear_cache()
        cat_latents = []
        
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = self.vae.encode(refs, device=self.device, tiled=tiled_vae)
                else:
                    ref_latent = self.vae.encode(refs, device=self.device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]

                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        
        return cat_latents

    def encode_masks(self, masks, ref_images=None):
        """Encode masks to latent space"""
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // VAE_STRIDE[0])
            height = 2 * (int(height) // (VAE_STRIDE[1] * 2))
            width = 2 * (int(width) // (VAE_STRIDE[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, VAE_STRIDE[1], width, VAE_STRIDE[1]
            )
            mask = mask.permute(2, 4, 0, 1, 3)
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                # Calculate latent temporal dimension for reference images
                # refs shape is [1, C, T, H, W]
                pixel_frames = refs.shape[2]
                latent_ref_length = (pixel_frames - 1) // VAE_STRIDE[0] + 1
                
                # Create zero mask for reference frames
                mask_pad = torch.zeros(
                    (mask.shape[0], latent_ref_length, mask.shape[2], mask.shape[3]),
                    device=mask.device,
                    dtype=mask.dtype
                )
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        
        return result_masks
    
    def combine_latent(self, z, m):
        """Combine latents and masks"""
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def decode_latent(self, zs, ref_images=None):
        """Decode latents back to frames"""
        return self.vae.decode(zs, device=self.device)

    def decode_single_frame(self, latent, index=-1):
        """Decode a specific frame index from a latent tensor [C, T, H, W]"""
        # latent is usually [C, T, H, W]
        # We need to wrap it for the VAE which expects [B, C, T, H, W] or list
        # FramePack VAE expects a list of tensors
        single_latent = latent[:, index:index+1, :, :].clone()
        return self.vae.decode([single_latent], device=self.device)


class ReferenceImageProcessor:
    """Handles reference image processing"""
    
    @staticmethod
    def process_reference_images(ref_images, width, height, device, dtype, target_ref_count=1):
        """Process reference images for generation"""
        
        # Handle duplication if requested
        if ref_images.shape[0] == 1 and target_ref_count > 1:
            print(f"Duplicating single reference image {target_ref_count} times")
            ref_images = ref_images.repeat(target_ref_count, 1, 1, 1)

        B, H, W, C = ref_images.shape
        current_aspect = W / H
        target_aspect = width / height
        
        if current_aspect > target_aspect:
            new_h = int(W / target_aspect)
            pad_h = (new_h - H) // 2
            padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], 
                            device=ref_images.device, dtype=ref_images.dtype)
            padded[:, pad_h:pad_h+H, :, :] = ref_images
            ref_images = padded
        elif current_aspect < target_aspect:
            new_w = int(H * target_aspect)
            pad_w = (new_w - W) // 2
            padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], 
                            device=ref_images.device, dtype=ref_images.dtype)
            padded[:, :, pad_w:pad_w+W, :] = ref_images
            ref_images = padded
            
        ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, 
                                "lanczos", "center")
        ref_images = ref_images.movedim(0, 1) # [C, T, H, W]
        ref_images = (ref_images.to(dtype).to(device) * 2 - 1)
        
        return [ref_images.unsqueeze(0)]
