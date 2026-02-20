"""
ComfyUI Node class definitions for FramePack video generation.
"""

import torch
from comfy import model_management as mm
import math
import gc
import time

from ..utils.benchmarking import BenchmarkManager
from ..utils.prompts import PromptHandler
from ..utils.vae import VAEProcessor
from ..sampler.generator import generate_with_framepack_multi


class WanVACEVideoFramepackSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["dpm++", "unipc", "euler", "deis", "lcm"], {"default": "unipc"}),
                "mode": (["moc", "sparse", "frame", "all"], {"default": "moc"}),
                "num_frames": ("INT", {"default": 81, "min": 41, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "force_offload": ("BOOLEAN", {"default": True}),
                "multi_prompts": ("STRING", {
                    "default": "A person walking in a park\nThe person starts jogging\nThe person runs faster\nThe person slows down to rest", 
                    "multiline": True
                }),
                "encode_prompts": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                "ref_images": ("IMAGE",),
                "input_frames": ("VIDEO",),
                "input_mask": ("MASK",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "text_embeds_list": ("LIST",),
                "wan_t5_model": ("WANTEXTENCODER",),
            }
        }
    
    RETURN_TYPES = ("LATENT", "VIDEO")
    RETURN_NAMES = ("samples", "decoded_video")
    FUNCTION = "process"
    CATEGORY = "framepackVACE"
    DESCRIPTION = "A sampler specifically for the FramePack algorithm for long video generation using hierarchical context."

    def __init__(self):
        self.benchmark_manager = BenchmarkManager()
        
        # Optimize CUDA performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def process(self, model, vae, steps, cfg, shift, seed, scheduler, mode,
                num_frames, width, height, force_offload, multi_prompts,
                encode_prompts=True, ref_images=None, input_frames=None, 
                input_mask=None, negative_prompt="", sigmas=None, 
                text_embeds_list=None, wan_t5_model=None):
        """Main processing function for ComfyUI with multi-prompt support"""
        
        enable_benchmarking = True
        benchmark_output_dir = "./benchmarks"
        
        text_encoder = wan_t5_model
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        # Extract model components
        model_obj = model.model
        model_wrapper = model_obj.diffusion_model
        
        # Setup VAE
        vae_processor = VAEProcessor(vae.to(device).to(torch.float32), device)
        
        # Ensure dimensions are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        # Calculate number of sections
        INITIAL_FRAMES = 81
        num_sections = 1 if num_frames <= INITIAL_FRAMES else math.ceil(num_frames / INITIAL_FRAMES)
        
        # Parse prompts
        section_prompts = PromptHandler.parse_multi_prompts(multi_prompts, num_sections)
        
        # Encode prompts
        if text_encoder is not None:
            print("Encoding prompts for each section...")
            section_text_embeds = []
            for i, prompt in enumerate(section_prompts):
                print(f"Encoding prompt {i+1}/{num_sections}: {prompt[:50]}...")
                text_embed = PromptHandler.encode_prompt_for_section(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    text_encoder=text_encoder,
                    device=device
                )
                section_text_embeds.append(text_embed)
        elif text_embeds_list:
            section_text_embeds = text_embeds_list
        else:
            raise ValueError("Either text encoder or pre-encoded embeddings required")
        
        # Determine modes to run
        modes_to_run = [mode] if mode != "all" else ["sparse", "frame", "moc"]
        generated_results = []
        
        for current_mode in modes_to_run:
            print(f"\n{'='*20}\nRunning Mode: {current_mode}\n{'='*20}")
            
            # Clear internal model caches to ensure no state leakage
            if hasattr(model_wrapper, 'teacache_state'):
                model_wrapper.teacache_state.clear_all()
            if hasattr(model_wrapper, 'magcache_state'):
                model_wrapper.magcache_state.clear_all()
            if hasattr(model_wrapper, 'block_mask'):
                model_wrapper.block_mask = None
            
            # Full memory cleanup cycle
            model_wrapper.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()
            model_wrapper.to(device)
            
            # Initialize benchmarking for this run
            if enable_benchmarking:
                self.benchmark_manager = BenchmarkManager()
                self.benchmark_manager.overall_start_time = time.time()
                self.benchmark_manager.generation_params = {
                    'num_frames': num_frames,
                    'width': width,
                    'height': height,
                    'steps': steps,
                    'cfg': cfg,
                    'scheduler': scheduler,
                    'mode': current_mode,
                    'seed': seed,
                }
        
            # Delegate to the extracted generation function
            latents, _ = generate_with_framepack_multi(
                model_wrapper=model_wrapper,
                vae_processor=vae_processor,
                benchmark_manager=self.benchmark_manager,
                section_text_embeds=section_text_embeds,
                section_prompts=section_prompts,
                input_frames=input_frames,
                input_masks=input_mask,
                ref_images=ref_images,
                width=width,
                height=height,
                num_frames=num_frames,
                shift=shift,
                scheduler_name=scheduler,
                mode=current_mode,
                steps=steps,
                cfg=cfg,
                seed=seed,
                sigmas=sigmas,
                device=device,
                offload_device=offload_device,
                force_offload=force_offload,
            )
            
            generated_results.append(latents)
            
            # Generate and save benchmark report
            if enable_benchmarking:
                report = self.benchmark_manager.generate_report(section_prompts)
                print("\n" + report)
                self.benchmark_manager.save_report(report, benchmark_output_dir)
        
        # Combine results if multiple
        if len(generated_results) > 1:
            final_latents = torch.stack(generated_results, dim=0)
        else:
            final_latents = generated_results[0].unsqueeze(0)

        return ({"samples": final_latents}, )
