import os
import sys
import gc
import math
import time
import random
import types
import logging
import traceback
from contextlib import contextmanager
from functools import partial
import time
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF
import torch
import torch.nn.functional as F
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm

from wan.text2video import (WanT2V, T5EncoderModel, WanVAE, shard_model, FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps, FlowUniPCMultistepScheduler)
from .modules.model import VaceWanModel
from ..utils.preprocessor import VaceVideoProcessor


class FramepackVace(WanT2V):
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.
        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),

            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating VaceWanModel from {checkpoint_dir}")
        self.model = VaceWanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward,
                                                            usp_dit_forward_vace)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            for block in self.model.vace_blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.model.forward_vace = types.MethodType(usp_dit_forward_vace, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

        self.vid_proc = VaceVideoProcessor(downsample=tuple([x * y for x, y in zip(config.vae_stride, self.patch_size)]),
            min_area=480 * 832,
            max_area=480 * 832,
            min_fps=self.config.sample_fps,
            max_fps=self.config.sample_fps,
            zero_start=True,
            seq_len=32760,
            keep_last=True)

    def vace_encode_frames(self, frames, ref_images, masks=None, vae=None):
        vae = self.vae if vae is None else vae
        if ref_images is None:
            ref_images = [None] * len(frames)
        else:
            assert len(frames) == len(ref_images)

        if masks is None:
            latents = vae.encode(frames)
        else:
            masks = [torch.where(m > 0.5, 1.0, 0.0) for m in masks]
            inactive = [i * (1 - m) + 0 * m for i, m in zip(frames, masks)]
            reactive = [i * m + 0 * (1 - m) for i, m in zip(frames, masks)]
            inactive = vae.encode(inactive)
            reactive = vae.encode(reactive)
            latents = [torch.cat((u, c), dim=0) for u, c in zip(inactive, reactive)]

        cat_latents = []
        for latent, refs in zip(latents, ref_images):
            if refs is not None:
                if masks is None:
                    ref_latent = vae.encode(refs)
                else:
                    ref_latent = vae.encode(refs)
                    ref_latent = [torch.cat((u, torch.zeros_like(u)), dim=0) for u in ref_latent]
                assert all([x.shape[1] == 1 for x in ref_latent])
                latent = torch.cat([*ref_latent, latent], dim=1)
            cat_latents.append(latent)
        return cat_latents

    def vace_encode_masks(self, masks, ref_images=None, vae_stride=None):
        vae_stride = self.vae_stride if vae_stride is None else vae_stride
        if ref_images is None:
            ref_images = [None] * len(masks)
        else:
            assert len(masks) == len(ref_images)

        result_masks = []
        for mask, refs in zip(masks, ref_images):
            c, depth, height, width = mask.shape
            new_depth = int((depth + 3) // vae_stride[0])
            height = 2 * (int(height) // (vae_stride[1] * 2))
            width = 2 * (int(width) // (vae_stride[2] * 2))

            # reshape
            mask = mask[0, :, :, :]
            mask = mask.view(
                depth, height, vae_stride[1], width, vae_stride[1]
            )  # depth, height, 8, width, 8
            mask = mask.permute(2, 4, 0, 1, 3)  # 8, 8, depth, height, width
            mask = mask.reshape(
                vae_stride[1] * vae_stride[2], depth, height, width
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

    def prepare_source(self, src_video, src_mask, src_ref_images, num_frames, image_size, device):
        area = image_size[0] * image_size[1]
        self.vid_proc.set_area(area)
        if area == 720*1280:
            self.vid_proc.set_seq_len(75600)
        elif area == 480*832:
            self.vid_proc.set_seq_len(32760)
        else:
            raise NotImplementedError(f'image_size {image_size} is not supported')

        image_size = (image_size[1], image_size[0])
        image_sizes = []
        for i, (sub_src_video, sub_src_mask) in enumerate(zip(src_video, src_mask)):
            if sub_src_mask is not None and sub_src_video is not None:
                src_video[i], src_mask[i], _, _, _ = self.vid_proc.load_video_pair(sub_src_video, sub_src_mask)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = src_mask[i].to(device)
                src_mask[i] = torch.clamp((src_mask[i][:1, :, :, :] + 1) / 2, min=0, max=1)
                image_sizes.append(src_video[i].shape[2:])
            elif sub_src_video is None:
                src_video[i] = torch.zeros((3, num_frames, image_size[0], image_size[1]), device=device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(image_size)
            else:
                src_video[i], _, _, _ = self.vid_proc.load_video(sub_src_video)
                src_video[i] = src_video[i].to(device)
                src_mask[i] = torch.ones_like(src_video[i], device=device)
                image_sizes.append(src_video[i].shape[2:])

        for i, ref_images in enumerate(src_ref_images):
            if ref_images is not None:
                image_size = image_sizes[i]
                for j, ref_img in enumerate(ref_images):
                    if ref_img is not None:
                        ref_img = Image.open(ref_img).convert("RGB")
                        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
                        if ref_img.shape[-2:] != image_size:
                            canvas_height, canvas_width = image_size
                            ref_height, ref_width = ref_img.shape[-2:]
                            white_canvas = torch.ones((3, 1, canvas_height, canvas_width), device=device) # [-1, 1]
                            scale = min(canvas_height / ref_height, canvas_width / ref_width)
                            new_height = int(ref_height * scale)
                            new_width = int(ref_width * scale)
                            resized_image = F.interpolate(ref_img.squeeze(1).unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0).unsqueeze(1)
                            top = (canvas_height - new_height) // 2
                            left = (canvas_width - new_width) // 2
                            white_canvas[:, :, top:top + new_height, left:left + new_width] = resized_image
                            ref_img = white_canvas
                        src_ref_images[i][j] = ref_img.to(device)

        return src_video, src_mask, src_ref_images

    def prepare_video(self, src_video):
        import decord
        decord.bridge.set_bridge('torch')
        reader = decord.VideoReader(src_video)

        total_frames = len(reader)
        fps = reader.get_avg_fps()
        num_frames=None
        # Get frame indices
        if num_frames is None:
            frame_ids = list(range(total_frames))
        else:
            # Sample frames evenly
            frame_ids = np.linspace(0, total_frames-1, num_frames, dtype=int).tolist()

        # Load frames
        video = reader.get_batch(frame_ids)  # [T, H, W, C]
        video = video.permute(3, 0, 1, 2)    # [C, T, H, W]

        # Convert to float and normalize
        video = video.float()
        C, T, H, W = video.shape
        chunk_length=81
        video = video / 255.0 
        C, T, H, W = video.shape
        usable_frames = (T // chunk_length) * chunk_length
        video = video[:, :usable_frames, :, :]  # Trim excess frames

        chunks = []
        for i in range(0, usable_frames, chunk_length):
            chunk = video[:, i:i+chunk_length, :, :]
            chunks.append(chunk)

        return chunks
    def decode_latent(self, zs, ref_images=None, vae=None):
        vae = self.vae if vae is None else vae

    # No need to check ref_images length or trim anymore
        return vae.decode(zs)


    def generate_with_framepack(self,
                 input_prompt,
                 input_frames,
                 input_masks,
                 input_ref_images,
                 
                 
                 size=(1280, 720),
                 frame_num=41,
                 context_scale=1,
                 shift=5.0,
                 sample_solver='dpm++',
                 sampling_steps=20,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        """
        Generates long videos using hierarchical context with frame packing.
        
        Major changes:
        1. Fixed context propagation between sections
        2. Improved hierarchical frame selection
        3. Better mask generation for consistent 22-frame structure
        4. Enhanced debugging and visualization
        """
        total_output_frames=0

        LATENT_WINDOW = 41  
        GENERATION_FRAMES = 30
        CONTEXT_FRAMES = 11
        INITIAL_FRAMES=121
        # frame_num=300
        section_window = 41 
        # section_num = math.ceil(frame_num / section_window)
        if frame_num <= INITIAL_FRAMES:
            section_num = 1
        else:
            remaining_frames = frame_num - INITIAL_FRAMES
            effective_step = GENERATION_FRAMES - CONTEXT_FRAMES
            section_num = 1 + math.ceil(remaining_frames / effective_step)


        all_generated_latents = []  
        accumulated_latents = []    
        context_buffer = None      

        print(f'Total frames requested: {frame_num}')
        print(f'Total sections to generate: {section_num}')
        print(f'Latent structure: {CONTEXT_FRAMES} context + {GENERATION_FRAMES} generation = {LATENT_WINDOW} total')

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # Base seed management
        if seed == -1:
            import time
            base_seed = int(time.time() * 1000) % (1 << 32)
        else:
            base_seed = seed
        self.model.to(self.device)
        for section_id in range(section_num):
            torch.cuda.synchronize()  
            print(f"\n{'='*60}")
            print(f"SECTION {section_id+1} / {section_num}")
            print(f"{'='*60}\n")
            
            def get_tensor_list_memory(tensor_list):
                total_bytes = 0
                for tensor in tensor_list:
                    if isinstance(tensor, torch.Tensor):
                        total_bytes += tensor.numel() * tensor.element_size()
                total_mb = total_bytes / (1024 ** 2)
                total_gb = total_bytes / (1024 ** 3)
                print(f"Total memory used by tensor list: {total_mb:.2f} MB ({total_gb:.4f} GB)")
                
            get_tensor_list_memory(accumulated_latents)
            get_tensor_list_memory(all_generated_latents)

            # Create unique seed for each section
            section_seed = base_seed + section_id * 1000
            section_generator = torch.Generator(device=self.device)
            section_generator.manual_seed(section_seed)


            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]


            if section_id == 0:

                print("First section - using input frames")

                current_frames=input_frames
                current_masks=input_masks
                current_ref_images = input_ref_images
                frame_offset = 0
                context_scale_section = context_scale

            else:

                print(f"Section {section_id} - building hierarchical context")


                context_latent = self.build_hierarchical_context_latent(
                    accumulated_latents, section_id)
                

                # context_decoded = self.decode_latent([context_latent], None)
               
                # get_tensor_list_memory(context_decoded)
                # self.model.to(self.device)

                if section_id > 1:
                    appearance, motion = self.separate_appearance_and_motion(context_latent)
                    motion_noise = torch.randn_like(motion) * 0.3
                    motion_perturbed = motion + motion_noise
                    context_decoded = [appearance + motion_perturbed * 0.5]


                hierarchical_frames = self.pick_context_v2(context_latent, section_id)
                current_frames = self.decode_latent([hierarchical_frames], None)
                print('current frames shape', current_frames[0].shape )
                current_masks = self.create_temporal_blend_mask_v2(
                    current_frames[0].shape, section_id)
                current_ref_images = None
                print('current mask shape', current_masks[0].shape )
                
                frame_offset = min(LATENT_WINDOW + (section_id - 1) * GENERATION_FRAMES, 100)


                context_variation = 0.7 + torch.rand(1).item() * 0.6
                context_scale_section = context_scale * context_variation
             
            z0 = self.vace_encode_frames(current_frames, current_ref_images, masks=current_masks)
            m0 = self.vace_encode_masks(current_masks, current_ref_images)
            z = self.vace_latent(z0, m0)
            print(f"Context latent shape: {z0[0].shape}")
            print(f"Context scale: {context_scale_section:.3f}")
            print(f"Frame offset: {frame_offset}")
        

            target_shape = list(z0[0].shape)
            target_shape[0] = int(target_shape[0] / 2)
            del z0,m0,current_frames,current_masks
            noise_base = torch.randn(
                    target_shape[0],
                    target_shape[1],
                    target_shape[2],
                    target_shape[3],
                    dtype=torch.float32,
                    device=self.device,
                    generator=section_generator)

            if section_id > 0 and accumulated_latents:
                noise = [noise_base]

            else:
                noise = [noise_base]

            print(f"Noise shape: {noise[0].shape}")

            seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size


            @contextmanager
            def noop_no_sync():
                yield
            sample_solver ='unipc'
            sampling_steps=50
            print(sample_solver, sampling_steps)
            no_sync = getattr(self.model, 'no_sync', noop_no_sync)
            # sample_solver='dpm++'
            # sampling_steps=20
            with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
                # Setup scheduler
                if sample_solver == 'dpm++':
                    sample_scheduler = FlowDPMSolverMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                    timesteps, _ = retrieve_timesteps(
                        sample_scheduler,
                        device=self.device,
                        sigmas=sampling_sigmas)
                elif sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                else:
                    raise NotImplementedError(f"Unsupported solver: {sample_solver}")

                latents = noise
                arg_c = {'context': context, 'seq_len': seq_len, 'frame_offset': frame_offset}
                arg_null = {'context': context_null, 'seq_len': seq_len, 'frame_offset': frame_offset}

                # Denoising loop
                for step_idx, t in enumerate(tqdm(timesteps, desc=f"Section {section_id+1}")):
                    latent_model_input = latents
                    timestep = torch.stack([t])

                   

                    noise_pred_cond = self.model(
                        latent_model_input, t=timestep, vace_context=z, 
                        vace_context_scale=context_scale_section, **arg_c)[0]
                    noise_pred_uncond = self.model(
                        latent_model_input, t=timestep, vace_context=z, 
                        vace_context_scale=context_scale_section, **arg_null)[0]

                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)


                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0),
                        t,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=section_generator)[0]
                    latents = [temp_x0.squeeze(0)]
                    
                    
                    # Debug first and last steps
                    if step_idx == 0 or step_idx == len(timesteps) - 1:
                        print(f"  Step {step_idx}: t={t.item():.3f}, "
                            f"latent stats: mean={latents[0].mean().item():.3f}, "
                            f"std={latents[0].std().item():.3f}")
                    del noise_pred, noise_pred_uncond,noise_pred_cond
                    
                    
            del context, context_null,z, noise
            if section_id == 0:
                print(f"Section 0: Removing {1} reference frames from latent")

                if section_num==1:
                    latent_without_ref = latents[0]
                    accumulated_latents.append(latent_without_ref)


                    all_generated_latents.append(latent_without_ref)
                    
                else: 
                    latent_without_ref = latents[0][:, 1:-10, :, :]
                    accumulated_latents.append(latent_without_ref)


                    all_generated_latents.append(latent_without_ref)
               
            else:
                if section_id > 2:
                    accumulated_latents.pop(0)
                new=latents[0][:, -GENERATION_FRAMES:, :, :]
                accumulated_latents.append(new)

                if section_id == 0:
                    # First section without reference images
                    all_generated_latents.append(latents[0])
                else:
                    # Take only newly generated frames
                    new_content = latents[0][:, -GENERATION_FRAMES:, :, :]
                    new_content = new_content[:, 11:, :, :]
                    all_generated_latents.append(new_content)
                    frames_added = new_content.shape[1]
                    total_output_frames += frames_added
                    print(f"🎬 Section {section_id} OUTPUT: Added {frames_added} frames to final video (took last {GENERATION_FRAMES}, removed first {CONTEXT_FRAMES})")
                    print(f"📊 Total output frames so far: {total_output_frames}")
                    del new_content
            torch.cuda.synchronize()    
            torch.cuda.empty_cache()
            gc.collect()

            # if self.rank == 0:
            #     section_decoded = self.decode_latent(latents, None)
            #     self.save_section_debug(section_decoded[0], section_id, accumulated_latents)

            # print(f"Section {section_id} completed. Generated latent shape: {latents[0].shape}")
        
        # Final video assembly
        if self.rank == 0 and all_generated_latents:
            if hasattr(self, 'model') and self.model is not None:
                self.model = self.model.cpu()
                print("  ✓ Moved DiT model to CPU")

            final_latent = torch.cat(all_generated_latents, dim=1)
            print(f"\nFinal latent shape: {final_latent.shape}")


            final_video = self.decode_latent([final_latent], None)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            
            if offload_model:
                gc.collect()
                torch.cuda.synchronize()

            if dist.is_initialized():
                dist.barrier()

            return final_video[0]

        return None
    
    def build_hierarchical_context_latent(self, accumulated_latents, section_id):
        """
        Build hierarchical context from accumulated latents.
        
        """
        if not accumulated_latents:
            raise ValueError("No accumulated latents available")

        all_prev = torch.cat(accumulated_latents, dim=1)
        total_frames = all_prev.shape[1]

        print(f"Building context from {total_frames} accumulated frames")

        return all_prev
    
    
    def pick_context_v2(self, frames, section_id, initial=False):
        """
        Enhanced hierarchical context selection with constant 22-frame output.
        
        Changes from original:
        1. Better handling of initial frames
        2. More robust frame selection with proper bounds checking
        3. Improved debugging output
        """

        # Constants
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

        if T >= 40:
            step = max(4, T // 20)
            long_indices = []
            for i in range(LONG_FRAMES):
                idx = min(i * step, T - 15)  
                long_indices.append(idx)
            selected_indices.extend(long_indices)
        else:
            # Not enough frames - take evenly spaced
            if T >= LONG_FRAMES:
                step = T // LONG_FRAMES
                long_indices = [i * step for i in range(LONG_FRAMES)]
            else:
                long_indices = list(range(T))
                # Pad by repeating last frame
                while len(long_indices) < LONG_FRAMES:
                    long_indices.append(T - 1)
            selected_indices.extend(long_indices[:LONG_FRAMES])

        mid_start = max(LONG_FRAMES, T - 15)
        mid_indices = [
            min(mid_start, T - 1),
            min(mid_start + 2, T - 1)
        ]
        selected_indices.extend(mid_indices)

        recent_idx = max(0, T - 5)
        selected_indices.append(recent_idx)

        overlap_start = max(0, T - OVERLAP_FRAMES)
        overlap_indices = list(range(overlap_start, T))

        while len(overlap_indices) < OVERLAP_FRAMES:
            overlap_indices.append(T - 1)
        selected_indices.extend(overlap_indices[:OVERLAP_FRAMES])
        context_frames = frames[:, selected_indices, :, :]

        gen_placeholder = torch.zeros((C, GEN_FRAMES, H, W), device=frames.device)

        final_frames = torch.cat([
            context_frames[:, :LONG_FRAMES],     
            context_frames[:, LONG_FRAMES:LONG_FRAMES+MID_FRAMES],  
            context_frames[:, LONG_FRAMES+MID_FRAMES:LONG_FRAMES+MID_FRAMES+RECENT_FRAMES], 
            context_frames[:, -OVERLAP_FRAMES:],  
            gen_placeholder                       
        ], dim=1)

        assert final_frames.shape[1] == TOTAL_FRAMES, \
            f"Expected {TOTAL_FRAMES} frames, got {final_frames.shape[1]}"

        if section_id % 5 == 0:
            print(f"\nContext selection debug (section {section_id}):")
            print(f"  Input frames: {T}")
            print(f"  Selected indices: {selected_indices}")
            print(f"  Output shape: {final_frames.shape}")

        return final_frames
    
    
    def create_temporal_blend_mask_v2(self, frame_shape, section_id, initial=False):
        """
        Enhanced mask creation that handles decoded frame dimensions
        """
        C, T, H, W = frame_shape
        LONG_FRAMES = 5
        MID_FRAMES = 3
        RECENT_FRAMES = 1
        OVERLAP_FRAMES = 2
        GEN_FRAMES = 30
        TOTAL_FRAMES = 41
        # Calculate the temporal expansion ratio
        LATENT_FRAMES = 41
        decoded_frames = T
        expansion_ratio = decoded_frames / LATENT_FRAMES
        
        mask = torch.zeros(1, decoded_frames, H, W, device=self.device)
        
        # Scale all frame counts by the expansion ratio
        LONG_FRAMES = int(5 * expansion_ratio)
        MID_FRAMES = int(3 * expansion_ratio)
        RECENT_FRAMES = int(1 * expansion_ratio)
        OVERLAP_FRAMES = int(2 * expansion_ratio)
        GEN_FRAMES = decoded_frames - (LONG_FRAMES + MID_FRAMES + RECENT_FRAMES + OVERLAP_FRAMES)
        
        if initial:
            mask[:, :-GEN_FRAMES] = 0.0  
            mask[:, -GEN_FRAMES:] = 1.0 
            return [mask]
        
        # Apply mask values with expanded frame counts
        idx = 0
        mask[:, idx:idx+LONG_FRAMES] = 0.05  
        idx += LONG_FRAMES
        
        mask[:, idx:idx+MID_FRAMES] = 0.2
        idx += MID_FRAMES
        
        mask[:, idx:idx+RECENT_FRAMES] = 0.3
        idx += RECENT_FRAMES
        
        for i in range(OVERLAP_FRAMES):
            blend_value = 0.4 + (i / (OVERLAP_FRAMES - 1)) * 0.4
            mask[:, idx+i] = blend_value
        idx += OVERLAP_FRAMES
        
        mask[:, idx:] = 1.0
        
        return [mask]
    def create_spatial_variation(self, H, W):
        """Create spatial variation mask for natural blending."""
        y_coords = torch.linspace(-1, 1, H, device=self.device)
        x_coords = torch.linspace(-1, 1, W, device=self.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')


        distance = torch.sqrt(x_grid**2 + y_grid**2) / 1.414  
        variation = 1.0 - 0.3 * torch.exp(-3 * distance**2)

        return variation

    def separate_appearance_and_motion(self, frames):
        """Use frequency domain to separate appearance from motion"""

        C, T, H, W = frames.shape


        fft_frames = torch.fft.rfft2(frames, dim=(-2, -1))


        fft_h = H
        fft_w = W // 2 + 1

        h_freqs = torch.fft.fftfreq(H, device=frames.device)

        w_freqs = torch.fft.rfftfreq(W, device=frames.device)


        h_grid, w_grid = torch.meshgrid(h_freqs, w_freqs, indexing='ij')


        freq_magnitude = torch.sqrt(h_grid**2 + w_grid**2)


        cutoff = 0.1  
        low_pass_mask = (freq_magnitude < cutoff).float().to(frames.device)


        if low_pass_mask.shape != fft_frames.shape[-2:]:
            print(f"Mask shape: {low_pass_mask.shape}, FFT shape: {fft_frames.shape}")

            low_pass_mask = low_pass_mask[:fft_h, :fft_w]


        while low_pass_mask.dim() < fft_frames.dim():
            low_pass_mask = low_pass_mask.unsqueeze(0)


        appearance_fft = fft_frames * low_pass_mask
        motion_fft = fft_frames * (1 - low_pass_mask)


        appearance = torch.fft.irfft2(appearance_fft, s=(H, W))
        motion = torch.fft.irfft2(motion_fft, s=(H, W))

        return appearance, motion


    def save_section_debug(self, video_tensor, section_id, accumulated_latents):
        """Enhanced debugging output with more information."""
        import imageio
        import numpy as np

        # Save video
        output_path = f"debug_section_{section_id:03d}.mp4"

        video_np = video_tensor.cpu().detach()
        video_np = (video_np + 1.0) / 2.0
        video_np = torch.clamp(video_np, 0.0, 1.0)
        video_np = video_np.permute(1, 2, 3, 0).numpy()
        video_np_uint8 = (video_np * 255).astype(np.uint8)

        imageio.mimsave(output_path, video_np_uint8, fps=12)

        # Save debug info
        debug_info = {
            'section_id': section_id,
            'video_shape': list(video_tensor.shape),
            'accumulated_latents': len(accumulated_latents),
            'total_latent_frames': sum(l.shape[1] for l in accumulated_latents)
        }

        import json
        with open(f"debug_section_{section_id:03d}.json", 'w') as f:
            json.dump(debug_info, f, indent=2)

        print(f"Saved debug output to {output_path}")

