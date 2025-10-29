import torch
import numpy as np
from comfy import model_management as mm
import math
import gc
from contextlib import contextmanager
from tqdm import tqdm
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from comfy.utils import load_torch_file, ProgressBar, common_upscale

from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .wanvideo.utils.basic_flowmatch import FlowMatchScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
from .wanvideo.utils.scheduling_flow_match_lcm import FlowMatchLCMScheduler


from einops import rearrange

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
class WanVACEVideoFramepackSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),  # Added VAE input
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["dpm++", "unipc"], {"default": "unipc"}),
                
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                
                "frame_num": ("INT", {"default": 81, "min": 41, "max": 1000, "step": 1}),
                "context_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "image_width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "image_height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                
                # Input media
                
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
            },
            "optional": {
                "src_ref_images": ("IMAGE", {"default": None}),
                "src_video": ("VIDEO", {"default": None}),
                "src_mask": ("MASK", {"default": None}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "end_step": ("INT", {"default": -1, "min": -1, "max": 10000, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("LATENT", "VIDEO")
    RETURN_NAMES = ("samples", "decoded_video")
    FUNCTION = "process"
    CATEGORY = "framepackVACE"
    DESCRIPTION = "A sampler specifically for the FramePack algorithm for long video generation using hierarchical context."

    def __init__(self):
        # FramePack constants
        self.LATENT_WINDOW = 41
        self.GENERATION_FRAMES = 30
        self.CONTEXT_FRAMES = 11
        self.INITIAL_FRAMES = 81
        self.vae_stride = [1, 8, 8]  # Default VAE stride
        
    def process(self, model, vae, steps, cfg, shift, seed, scheduler, text_embeds, frame_num, 
                context_scale, image_width, image_height, 
                force_offload, src_ref_images=None,src_video=None, src_mask=None,  negative_prompt="", 
                start_step=0, end_step=-1):
        device = mm.get_torch_device()
        patcher = model
        model = model.model
        model_wrapper = model.diffusion_model
        dtype = model["dtype"]
        self.vae = vae 
        self.device=device    
        
        try:
            # Ensure model is on the right device
            
            
            if hasattr(vae, 'dtype'):
                self.vae_dtype = vae.dtype
            elif hasattr(vae, 'model') and hasattr(vae.model, 'dtype'):
                self.vae_dtype = vae.model.dtype
            else:
                # Fallback - try to infer from model parameters
                try:
                    self.vae_dtype = next(vae.parameters()).dtype
                except (StopIteration, AttributeError):
                    self.vae_dtype = torch.float32  # Final fallback
            if hasattr(self.vae, 'model'):
                self.vae.model = self.vae.model.to(device)
            else:
                self.vae = self.vae.to(device)
            print(f"Using VAE dtype: {self.vae_dtype}")
            
            if model_wrapper is None:
                raise ValueError("Invalid model provided")
            
            print(f"Starting FramePack generation: {frame_num} frames, {image_width}x{image_height}")
            
            # Prepare inputs using the original format
            height=512
            width= 512
            input_frames = self._prepare_video_input(src_video, device, frame_num, height, width)
            input_masks = self._prepare_mask_input(src_mask, device,  frame_num, height, width)
            input_ref_images = self._prepare_ref_images(src_ref_images, device) if src_ref_images is not None else None
            
            # Extract text embeddings
            if text_embeds is None:
                text_embeds = {
                    "prompt_embeds": [],
                    "negative_prompt_embeds": [],
                }
            
            positive_embeds = text_embeds.get('prompt_embeds', None)
            if positive_embeds is None:
                # Try alternative key names
                positive_embeds = text_embeds.get('positive', None)
            
            negative_embeds = text_embeds.get('negative_prompt_embeds', None)
            if negative_embeds is None:
                # Try alternative key names
                negative_embeds = text_embeds.get('negative', None)
            
            # Handle custom negative prompt
            if negative_prompt != "" and negative_prompt is not None:
                negative_embeds = self._encode_negative_prompt(negative_prompt, model)
            
            if positive_embeds is None or len(positive_embeds) == 0:
                raise ValueError("No positive text embeddings provided. Please connect a text encoder node.")
            
            
            # Generate video using FramePack algorithm
            generated_video = self._generate_with_framepack(
                model_wrapper=model_wrapper,
                positive_embeds=positive_embeds,
                negative_embeds=negative_embeds,
                input_frames=input_frames,
                input_masks=input_masks,
                ref_images=input_ref_images,
                size=(image_width, image_height),
                num_frames=frame_num,
                context_scale=context_scale,
                shift=shift,
                sample_solver=scheduler,
                sampling_steps=steps,
                guide_scale=cfg,
                seed=seed,
                offload_model=force_offload,
                device=device
            )
            
            # Convert to ComfyUI latent format
            latent_samples = self._video_to_latent_format(generated_video)
            
            print("FramePack generation completed successfully")
            
            return (latent_samples, generated_video)
            
        except Exception as e:
            print(f"Error in FramePack generation: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def _generate_with_framepack(self, model_wrapper, positive_embeds, negative_embeds,
                                input_frames, input_masks, ref_images,
                                size, num_frames, context_scale, shift, sample_solver,
                                sampling_steps, guide_scale, seed, offload_model, device):
        """
        Main FramePack generation method adapted from your implementation
        """
        
        # Calculate sections
        if num_frames <= self.INITIAL_FRAMES:
            section_num = 1
        else:
            remaining_frames = num_frames - self.INITIAL_FRAMES
            effective_step = self.GENERATION_FRAMES - self.CONTEXT_FRAMES
            section_num = 1 + math.ceil(remaining_frames / effective_step)

        all_generated_latents = []
        accumulated_latents = []
        
        print(f'Total frames requested: {num_frames}')
        print(f'Total sections to generate: {section_num}')
        print(f'Latent structure: {self.CONTEXT_FRAMES} context + {self.GENERATION_FRAMES} generation = {self.LATENT_WINDOW} total')

        # Base seed management
        if seed == -1:
            import time
            base_seed = int(time.time() * 1000) % (1 << 32)
        else:
            base_seed = seed

        # Move model to device
        model_wrapper.to(device)

        for section_id in range(section_num):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            print(f"Processing Section {section_id+1} / {section_num}")
            
            # Create unique seed for each section
            section_seed = base_seed + section_id * 1000
            section_generator = torch.Generator(device=device)
            section_generator.manual_seed(section_seed)
            height=512
            width= 512
            input_frames=None
            input_masks=None
            # Prepare context for this section
            if section_id == 0:
                width = (width // 16) * 16
                height = (height // 16) * 16

                target_shape = (16, (num_frames - 1) // VAE_STRIDE[0] + 1,
                                height // VAE_STRIDE[1],
                                width // VAE_STRIDE[2])
                # vace context encode
                if input_frames is None:
                    input_frames = torch.zeros((1, 3, num_frames, height, width), device=self.device, dtype=self.vae.dtype)
                else:
                    input_frames = input_frames[:num_frames]
                    input_frames = input_frames.to(torch.float32)
                    input_frames = common_upscale(
                        input_frames.clone().movedim(-1, 1),
                        width, height, "lanczos", "disabled"
                    ).movedim(1, -1)
                    input_frames = input_frames.to(self.vae.dtype).to(self.device).unsqueeze(0).permute(0, 4, 1, 2, 3) # B, C, T, H, W
                    input_frames = input_frames * 2 - 1
                if input_masks is None:
                    input_masks = torch.ones_like(input_frames, device=self.device)
                else:
                    print("input_masks shape", input_masks.shape)
                    input_masks = input_masks[:num_frames]
                    input_masks = common_upscale(input_masks.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
                    input_masks = input_masks.to(self.vae.dtype).to(self.device)
                    input_masks = input_masks.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1) # B, C, T, H, W

                if ref_images is not None:
                    # Create padded image
                    if ref_images.shape[0] > 1:
                        ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], dim=1).unsqueeze(0)
                
                    B, H, W, C = ref_images.shape
                    current_aspect = W / H
                    target_aspect = width / height
                    if current_aspect > target_aspect:
                        # Image is wider than target, pad height
                        new_h = int(W / target_aspect)
                        pad_h = (new_h - H) // 2
                        padded = torch.ones(ref_images.shape[0], new_h, W, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                        padded[:, pad_h:pad_h+H, :, :] = ref_images
                        ref_images = padded
                    elif current_aspect < target_aspect:
                        # Image is taller than target, pad width
                        new_w = int(H * target_aspect)
                        pad_w = (new_w - W) // 2
                        padded = torch.ones(ref_images.shape[0], H, new_w, ref_images.shape[3], device=ref_images.device, dtype=ref_images.dtype)
                        padded[:, :, pad_w:pad_w+W, :] = ref_images
                        ref_images = padded
                    ref_images = common_upscale(ref_images.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
                    
                    ref_images = ref_images.to(self.vae.dtype).to(self.device).unsqueeze(0).permute(0, 4, 1, 2, 3).unsqueeze(0)
                    ref_images = ref_images * 2 - 1
                print("First section - using input frames")
                current_frames = input_frames
                current_masks = input_masks
                current_ref_images = ref_images
                frame_offset = 0
                context_scale_section = context_scale
            else:
                print(f"Section {section_id} - building hierarchical context")
                
                # Build hierarchical context
                context_latent = self._build_hierarchical_context_latent(accumulated_latents, section_id)
                
                # Pick context frames using hierarchical selection
                hierarchical_frames = self._pick_context_v2(context_latent, section_id)
                current_frames = self.decode_latent([hierarchical_frames], ref_images=None, vae=self.vae)[0]
                current_masks = self._create_temporal_blend_mask_v2(current_frames.shape, section_id, device)
                current_ref_images = None
                
                frame_offset = min(self.LATENT_WINDOW + (section_id - 1) * self.GENERATION_FRAMES, 100)
                
                # Add context variation
                context_variation = 0.7 + torch.rand(1).item() * 0.6
                context_scale_section = context_scale * context_variation
            # current_frames=  current_frames[0]
            # current_masks= current_masks[0]
            # current_ref_images= current_ref_images[0].unsqueeze(0)
            if current_ref_images is not None:
                # Handle the case where ref_images might be a single tensor or already a list
                if not isinstance(current_ref_images, list):
                    ref_images_list = [current_ref_images]
                else:
                    ref_images_list = current_ref_images
            else:
                ref_images_list = [None]
            print("current_frames[0] shape:", current_frames[0].shape)
            print("current_ref_images[0] shape:", ref_images_list[0].shape)
            print("current_masks[0] shape:", current_masks[0].shape)
            
            
            if current_frames is not None:
                current_frames = current_frames.to(self.device)
            if current_masks is not None:  
                current_masks = current_masks.to(self.device)
            if ref_images_list[0] is not None:
                ref_images_list = [r.to(self.device) if r is not None else None for r in ref_images_list]
           
            # Encode frames to latent space input_ref_imagesusing VACE methods
            z0 = self.vace_encode_frames(input_frames, ref_images, masks=input_masks, tiled_vae=False)
            self.vae.model.clear_cache()
            m0 = self.vace_encode_masks(input_masks, ref_images)
            z = self.vace_latent(z0, m0)
            
            print(f"Context latent shape: {z0.shape if hasattr(z0, 'shape') else 'N/A'}")
            print(f"Context scale: {context_scale_section:.3f}")
            print(f"Frame offset: {frame_offset}")

            # Prepare noise
            target_shape = list(z0.shape) if hasattr(z0, 'shape') else [1, self.LATENT_WINDOW // 2, 64, 64]
            if len(target_shape) > 0:
                target_shape[0] = int(target_shape[0] / 2)
            
            noise_base = torch.randn(
                target_shape,
                dtype=torch.float32,
                device=device,
                generator=section_generator
            )
            

            print(f"Noise shape: {noise[0].shape}")

            # Setup scheduler
            scheduler_obj = self._setup_scheduler(sample_solver, sampling_steps, shift, device)
            timesteps = scheduler_obj.timesteps
            seq_len=32760
            freqs = None
            model_wrapper.rope_embedder.k = None
            model_wrapper.rope_embedder.num_frames = None
            latent_video_length = noise.shape[1]
            riflex_freq_index=0
            if  True:
                d = model_wrapper.dim // model_wrapper.num_heads
                freqs = torch.cat([
                    rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6))
                ],
                dim=1)
            # Denoising loop
            noise = [noise_base]
            latents = noise
            with torch.no_grad():
                for step_idx, t in enumerate(tqdm(timesteps, desc=f"Section {section_id+1}")):
                    timestep = torch.stack([t])
                    base_params = {
                    'seq_len': seq_len,
                    'device': device,
                    'freqs': freqs,
                    't': timestep,
                    'current_step': step_idx,
                    
                    "nag_params": text_embeds.get("nag_params", {}),
                    "nag_context": text_embeds.get("nag_prompt_embeds", None),
                    
                }


                    # Get noise predictions
                    noise_pred_cond = model_wrapper(
                        latents, t=timestep, 
                        vace_data=[
                    {"context":z, 
                     "scale": context_scale_section, 
                   
                     }
                ],
                        vace_context_scale=context_scale_section,
                        context=positive_embeds,
                        **base_params
                    )[0]
                    
                    noise_pred_uncond = model_wrapper(
                        latents, t=timestep,
                        vace_data=[
                    {"context":z, 
                     "scale": context_scale_section, 
                   
                     }
                ],               
                        context=positive_embeds,
                        **base_params
                    )[0]

                    # Apply CFG
                    noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

                    # Scheduler step
                    temp_x0 = scheduler_obj.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=section_generator
                    )[0]
                    latents = [temp_x0.squeeze(0)]

                    # Debug output
                    if step_idx == 0 or step_idx == len(timesteps) - 1:
                        print(f"  Step {step_idx}: t={t.item():.3f}, latent stats: mean={latents[0].mean().item():.3f}, std={latents[0].std().item():.3f}")

            # Process generated latents
            if section_id == 0:
                if section_num == 1:
                    accumulated_latents.append(latents[0])
                    all_generated_latents.append(latents[0])
                else:
                    latent_without_ref = latents[0][:, 1:-10, :, :]
                    accumulated_latents.append(latent_without_ref)
                    all_generated_latents.append(latent_without_ref)
            else:
                # Remove old context if we have too many sections
                if section_id > 2:
                    accumulated_latents.pop(0)
                
                new = latents[0][:, -self.GENERATION_FRAMES:, :, :]
                accumulated_latents.append(new)
                
                # Take only newly generated frames for output
                new_content = latents[0][:, -self.GENERATION_FRAMES:, :, :]
                new_content = new_content[:, self.CONTEXT_FRAMES:, :, :]
                all_generated_latents.append(new_content)
                
                print(f"Section {section_id} OUTPUT: Added {new_content.shape[1]} frames")

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Final video assembly
        if all_generated_latents:
            if offload_model and hasattr(model_wrapper, 'cpu'):
                model_wrapper.cpu()
                print("Moved model to CPU")

            final_latent = torch.cat(all_generated_latents, dim=1)
            print(f"Final latent shape: {final_latent.shape}")

            # Decode final video using VACE decoder
            final_video = self.decode_latent([final_latent], ref_images=None, vae=self.vae)[0]

            return final_video

        return None

    
    def vace_encode_frames(self, frames, ref_images, masks=None, tiled_vae=False):
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
                    print("refs shape", refs.shape)#torch.Size([3, 1, 512, 512])
                    ref_latent = self.vae.encode(refs, device=self.device, tiled=tiled_vae)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None):
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
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                VAE_STRIDE[1] * VAE_STRIDE[2], depth, height, width
            )  # 8*8, depth, height, width

            # interpolation
            mask = F.interpolate(mask.unsqueeze(0), size=(new_depth, height, width), mode='nearest-exact').squeeze(0)

            if refs is not None:
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        return result_masks

    def vace_latent(self, z, m):
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def decode_latent(self, zs, ref_images=None, vae=None):
        vae = self.vae if vae is None else vae
        return vae.decode(zs)

    def _build_hierarchical_context_latent(self, accumulated_latents, section_id):
        """Build hierarchical context from accumulated latents"""
        if not accumulated_latents:
            raise ValueError("No accumulated latents available")
        
        all_prev = torch.cat(accumulated_latents, dim=1)
        print(f"Building context from {all_prev.shape[1]} accumulated frames")
        return all_prev

    def _pick_context_v2(self, frames, section_id, initial=False):
        """Enhanced hierarchical context selection with constant frame output"""
        LONG_FRAMES = 5
        MID_FRAMES = 3
        RECENT_FRAMES = 1
        OVERLAP_FRAMES = 2
        GEN_FRAMES = 30
        TOTAL_FRAMES = 41

        C, T, H, W = frames.shape

        if initial and T == TOTAL_FRAMES:
            return frames

        if initial and T < TOTAL_FRAMES:
            padding_needed = TOTAL_FRAMES - T
            padding = torch.zeros((C, padding_needed, H, W), device=frames.device)
            return torch.cat([frames, padding], dim=1)

        selected_indices = []

        # Long-term context
        if T >= 40:
            step = max(4, T // 20)
            long_indices = [min(i * step, T - 15) for i in range(LONG_FRAMES)]
        else:
            if T >= LONG_FRAMES:
                step = T // LONG_FRAMES
                long_indices = [i * step for i in range(LONG_FRAMES)]
            else:
                long_indices = list(range(T))
                while len(long_indices) < LONG_FRAMES:
                    long_indices.append(T - 1)
        
        selected_indices.extend(long_indices[:LONG_FRAMES])

        # Mid-term context
        mid_start = max(LONG_FRAMES, T - 15)
        mid_indices = [min(mid_start, T - 1), min(mid_start + 2, T - 1)]
        selected_indices.extend(mid_indices)

        # Recent context
        recent_idx = max(0, T - 5)
        selected_indices.append(recent_idx)

        # Overlap frames
        overlap_start = max(0, T - OVERLAP_FRAMES)
        overlap_indices = list(range(overlap_start, T))
        while len(overlap_indices) < OVERLAP_FRAMES:
            overlap_indices.append(T - 1)
        selected_indices.extend(overlap_indices[:OVERLAP_FRAMES])

        # Select context frames
        context_frames = frames[:, selected_indices, :, :]

        # Add generation placeholder
        gen_placeholder = torch.zeros((C, GEN_FRAMES, H, W), device=frames.device)

        final_frames = torch.cat([
            context_frames[:, :LONG_FRAMES],
            context_frames[:, LONG_FRAMES:LONG_FRAMES+MID_FRAMES],
            context_frames[:, LONG_FRAMES+MID_FRAMES:LONG_FRAMES+MID_FRAMES+RECENT_FRAMES],
            context_frames[:, -OVERLAP_FRAMES:],
            gen_placeholder
        ], dim=1)

        assert final_frames.shape[1] == TOTAL_FRAMES
        return final_frames

    def _create_temporal_blend_mask_v2(self, frame_shape, section_id, device, initial=False):
        """Enhanced mask creation for temporal blending"""
        C, T, H, W = frame_shape
        
        # Calculate frame distribution
        LATENT_FRAMES = 41
        expansion_ratio = T / LATENT_FRAMES
        
        mask = torch.zeros(1, T, H, W, device=device)
        
        # Scale frame counts
        LONG_FRAMES = int(5 * expansion_ratio)
        MID_FRAMES = int(3 * expansion_ratio)
        RECENT_FRAMES = int(1 * expansion_ratio)
        OVERLAP_FRAMES = int(2 * expansion_ratio)
        GEN_FRAMES = T - (LONG_FRAMES + MID_FRAMES + RECENT_FRAMES + OVERLAP_FRAMES)
        
        if initial:
            mask[:, :-GEN_FRAMES] = 0.0
            mask[:, -GEN_FRAMES:] = 1.0
            return mask
        
        # Apply progressive masking
        idx = 0
        mask[:, idx:idx+LONG_FRAMES] = 0.05
        idx += LONG_FRAMES
        
        mask[:, idx:idx+MID_FRAMES] = 0.2
        idx += MID_FRAMES
        
        mask[:, idx:idx+RECENT_FRAMES] = 0.3
        idx += RECENT_FRAMES
        
        # Overlap with gradient
        for i in range(OVERLAP_FRAMES):
            blend_value = 0.4 + (i / max(1, OVERLAP_FRAMES - 1)) * 0.4
            mask[:, idx+i] = blend_value
        idx += OVERLAP_FRAMES
        
        # Full generation mask
        mask[:, idx:] = 1.0
        
        return mask

    # Helper methods for model operations
    def _prepare_video_input(self, video_input, device, num_frames, height, width):
        """Convert ComfyUI video input to tensor format or create default if None"""
        target_dtype = getattr(self, 'vae_dtype', torch.float32)
        if video_input is None:
            
            input_frames = torch.zeros((1, 3, num_frames, height, width), device=device, dtype=target_dtype)
            return input_frames
        else:
            # Process existing video input
            if hasattr(video_input, 'shape'):
                if len(video_input.shape) == 4:
                    # Expected format: [T, H, W, C] -> [B, C, T, H, W]
                    video_input = video_input[:num_frames]  # Limit to requested frames
                    # Resize if needed
                    from comfy.utils import common_upscale
                    video_input = common_upscale(video_input.clone().movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
                    video_input = video_input.to(target_dtype).to(device).unsqueeze(0).permute(0, 4, 1, 2, 3)  # B, C, T, H, W
                    video_input = video_input * 2 - 1  # Normalize to [-1, 1]
                    return video_input
            return video_input.to(device)

    def _prepare_mask_input(self, mask_input, device, num_frames, height, width):
        """Convert ComfyUI mask input to tensor format or create default if None"""
        if mask_input is None:
            target_dtype = getattr(self, 'vae_dtype', torch.float32)
            input_masks = torch.ones((1, 3, num_frames, height, width), device=device, dtype=target_dtype)
            return input_masks
        else:
            # Process existing mask input
            mask_input = mask_input[:num_frames]  # Limit to requested frames
            from comfy.utils import common_upscale
            mask_input = common_upscale(mask_input.clone().unsqueeze(1), width, height, "nearest-exact", "disabled").squeeze(1)
            mask_input = mask_input.to(self.vae.dtype).to(device)
            mask_input = mask_input.unsqueeze(-1).unsqueeze(0).permute(0, 4, 1, 2, 3).repeat(1, 3, 1, 1, 1)  # B, C, T, H, W
            return mask_input


    def _prepare_ref_images(self, ref_images, device):
        """Convert reference images to tensor format"""
        if ref_images is not None:
            return ref_images.to(device)
        return ref_images

    def _setup_scheduler(self, solver_name, steps, shift, device):
        """Setup the appropriate scheduler"""
        if solver_name == "dpm++":
            
            scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False
            )
        elif solver_name == "unipc":
            
            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False
            )
            scheduler.set_timesteps(steps, device=device, shift=shift)
        else:
            raise NotImplementedError(f"Unsupported solver: {solver_name}")
        
        return scheduler

    def _encode_negative_prompt(self, negative_prompt, model):
        """Encode negative prompt if provided"""
        # This needs to be implemented based on your text encoder
        return model.encode_text(negative_prompt)

    def _video_to_latent_format(self, video):
        """Convert video to ComfyUI latent format"""
        # Convert to the expected latent format for ComfyUI
        return {"samples": video}


# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": WanVACEVideoFramepackSampler2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": "WanVACE FramePack Sampler 2"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']