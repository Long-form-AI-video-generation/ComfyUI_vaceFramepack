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
from .utils import log, print_memory, apply_lora, clip_encode_image_tiled, fourier_filter

from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .wanvideo.utils.basic_flowmatch import FlowMatchScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
from .wanvideo.utils.scheduling_flow_match_lcm import FlowMatchLCMScheduler
import time
import psutil


from datetime import datetime
import json
import os

from einops import rearrange

VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)
class WanVACEVideoFramepackSampler2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "vae": ("WANVAE",),
                # Remove the single positiveprompts input
                # "positiveprompts": ("POSITIVEPROMPT"),
                "steps": ("INT", {"default": 30, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scheduler": (["dpm++", "unipc", "euler", "deis", "lcm"], {"default": "unipc"}),
                # Remove single text_embeds, will be handled per section
                # "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "num_frames": ("INT", {"default": 81, "min": 41, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 832, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 480, "min": 64, "max": 2048, "step": 8}),
                "force_offload": ("BOOLEAN", {"default": True}),
                # Add multi-prompt support
                "multi_prompts": ("STRING", {
                    "default": "A person walking in a park\nThe person starts jogging\nThe person runs faster\nThe person slows down to rest", 
                    "multiline": True
                }),
                "encode_prompts": ("BOOLEAN", {"default": True}),  # Whether to encode prompts internally
            },
            "optional": {
                "sigmas": ("SIGMAS",),
                "ref_images": ("IMAGE",),
                "input_frames": ("VIDEO",),
                "input_mask": ("MASK",),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                # Optional: pre-encoded embeddings for each section
                "text_embeds_list": ("LIST",),  # List of WANVIDEOTEXTEMBEDS
                "wan_t5_model": ("WANTEXTENCODER",),  # Optional text encoder
            }
        }
    RETURN_TYPES = ("LATENT", "VIDEO")
    RETURN_NAMES = ("samples", "decoded_video")
    FUNCTION = "process"
    CATEGORY = "framepackVACE"
    DESCRIPTION = "A sampler specifically for the FramePack algorithm for long video generation using hierarchical context."

    def __init__(self):
        self.LATENT_WINDOW = 41
        self.GENERATION_FRAMES = 30
        self.CONTEXT_FRAMES = 11
        self.INITIAL_FRAMES = 81
        self.cache_state = None
        self.benchmark_data = {}
        
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        
    import psutil
    import torch
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetUtilizationRates
    )

    def get_memory_stats(self):
        """Get current memory statistics"""
        stats = {}

        # CPU memory (process)
        process = psutil.Process()
        stats["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024
        stats["cpu_memory_percent"] = process.memory_percent()

        # System memory
        mem = psutil.virtual_memory()
        stats["system_memory_total_gb"] = mem.total / 1024**3
        stats["system_memory_used_gb"] = mem.used / 1024**3
        stats["system_memory_percent"] = mem.percent

        # GPU stats (if CUDA available)
        if torch.cuda.is_available():
            stats["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1024**3
            stats["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / 1024**3

            try:
                nvmlInit()
                device_count = nvmlDeviceGetCount()
                if device_count > 0:
                    handle = nvmlDeviceGetHandleByIndex(0)  # first GPU
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    stats["gpu_memory_total_gb"] = mem_info.total / 1024**3
                    stats["gpu_memory_used_gb"] = mem_info.used / 1024**3
                    util = nvmlDeviceGetUtilizationRates(handle)
                    stats["gpu_utilization_percent"] = util.gpu
            except Exception as e:
                print(f"GPU stats error: {e}")
            finally:
                try:
                    nvmlShutdown()
                except:
                    pass
        else:
            stats.update({
                "gpu_memory_allocated_gb": 0,
                "gpu_memory_reserved_gb": 0,
                "gpu_memory_total_gb": 0,
                "gpu_memory_used_gb": 0,
                "gpu_utilization_percent": 0,
            })

        return stats

    def benchmark_section(self, section_id, phase_name):
        """Start or end benchmarking for a section phase"""
        if not hasattr(self, 'section_benchmarks'):
            self.section_benchmarks = {}
        
        if section_id not in self.section_benchmarks:
            self.section_benchmarks[section_id] = {}
        
        phase_key = f"{phase_name}_start"
        phase_end_key = f"{phase_name}_end"
        
        if phase_key not in self.section_benchmarks[section_id]:
            # Starting phase
            self.section_benchmarks[section_id][phase_key] = time.time()
            self.section_benchmarks[section_id][f"{phase_name}_memory_start"] = self.get_memory_stats()
        else:
            # Ending phase
            self.section_benchmarks[section_id][phase_end_key] = time.time()
            self.section_benchmarks[section_id][f"{phase_name}_memory_end"] = self.get_memory_stats()
            
            # Calculate duration
            duration = self.section_benchmarks[section_id][phase_end_key] - self.section_benchmarks[section_id][phase_key]
            self.section_benchmarks[section_id][f"{phase_name}_duration"] = duration
            
            # Calculate memory delta
            start_mem = self.section_benchmarks[section_id][f"{phase_name}_memory_start"]
            end_mem = self.section_benchmarks[section_id][f"{phase_name}_memory_end"]
            
            if 'gpu_memory_allocated_gb' in start_mem and 'gpu_memory_allocated_gb' in end_mem:
                gpu_delta = end_mem['gpu_memory_allocated_gb'] - start_mem['gpu_memory_allocated_gb']
                self.section_benchmarks[section_id][f"{phase_name}_gpu_memory_delta_gb"] = gpu_delta
            
            cpu_delta = end_mem['cpu_memory_mb'] - start_mem['cpu_memory_mb']
            self.section_benchmarks[section_id][f"{phase_name}_cpu_memory_delta_mb"] = cpu_delta
    
    def generate_benchmark_report(self):
        """Generate a comprehensive benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("FRAMEPACK VIDEO GENERATION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if hasattr(self, 'overall_start_time'):
            total_duration = time.time() - self.overall_start_time
            report.append(f"Total Processing Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        if hasattr(self, 'generation_params'):
            report.append("\nGeneration Parameters:")
            for key, value in self.generation_params.items():
                report.append(f"  {key}: {value}")
        
        report.append("\n" + "-" * 80)
        report.append("SECTION-BY-SECTION BREAKDOWN")
        report.append("-" * 80)
        
        if hasattr(self, 'section_benchmarks'):
            total_encoding_time = 0
            total_denoising_time = 0
            total_accumulation_time = 0
            
            for section_id in sorted(self.section_benchmarks.keys()):
                section_data = self.section_benchmarks[section_id]
                report.append(f"\n[Section {section_id + 1}]")
                
                # Prompt info
                if hasattr(self, 'section_prompts') and section_id < len(self.section_prompts):
                    report.append(f"Prompt: {self.section_prompts[section_id][:50]}...")
                
                # Timing breakdown
                phases = ['encoding', 'denoising', 'accumulation']
                for phase in phases:
                    if f"{phase}_duration" in section_data:
                        duration = section_data[f"{phase}_duration"]
                        report.append(f"  {phase.capitalize()}: {duration:.2f}s")
                        
                        if phase == 'encoding':
                            total_encoding_time += duration
                        elif phase == 'denoising':
                            total_denoising_time += duration
                        elif phase == 'accumulation':
                            total_accumulation_time += duration
                        
                        # Memory changes
                        if f"{phase}_gpu_memory_delta_gb" in section_data:
                            gpu_delta = section_data[f"{phase}_gpu_memory_delta_gb"]
                            report.append(f"    GPU Memory Δ: {gpu_delta:+.3f} GB")
                        
                        if f"{phase}_cpu_memory_delta_mb" in section_data:
                            cpu_delta = section_data[f"{phase}_cpu_memory_delta_mb"]
                            report.append(f"    CPU Memory Δ: {cpu_delta:+.1f} MB")
                
                # Per-section total
                section_total = sum([section_data.get(f"{p}_duration", 0) for p in phases])
                report.append(f"  Section Total: {section_total:.2f}s")
                
                # Peak memory for section
                if 'denoising_memory_end' in section_data:
                    end_mem = section_data['denoising_memory_end']
                    if 'gpu_memory_allocated_gb' in end_mem:
                        report.append(f"  Peak GPU Memory: {end_mem['gpu_memory_allocated_gb']:.3f} GB")
                    report.append(f"  Peak CPU Memory: {end_mem['cpu_memory_mb']:.1f} MB")
        
        # Summary statistics
        report.append("\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)
        
        if hasattr(self, 'section_benchmarks'):
            num_sections = len(self.section_benchmarks)
            report.append(f"Total Sections Processed: {num_sections}")
            report.append(f"Total Encoding Time: {total_encoding_time:.2f}s")
            report.append(f"Total Denoising Time: {total_denoising_time:.2f}s")
            report.append(f"Total Accumulation Time: {total_accumulation_time:.2f}s")
            
            if num_sections > 0:
                report.append(f"Average Time per Section: {(total_encoding_time + total_denoising_time + total_accumulation_time) / num_sections:.2f}s")
                report.append(f"Average Denoising Time per Section: {total_denoising_time / num_sections:.2f}s")
        
        # Final memory state
        final_memory = self.get_memory_stats()
        report.append(f"\nFinal Memory State:")
        if 'gpu_memory_allocated_gb' in final_memory:
            report.append(f"  GPU Memory: {final_memory['gpu_memory_allocated_gb']:.3f} GB allocated")
        report.append(f"  CPU Memory: {final_memory['cpu_memory_mb']:.1f} MB")
        report.append(f"  System Memory: {final_memory['system_memory_percent']:.1f}% used")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_benchmark_report(self, report, output_dir="./benchmarks"):
        """Save benchmark report to file"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"framepack_benchmark_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        # Also save as JSON for easier analysis
        json_filename = f"framepack_benchmark_{timestamp}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        
        benchmark_dict = {
            'timestamp': timestamp,
            'generation_params': getattr(self, 'generation_params', {}),
            'section_benchmarks': getattr(self, 'section_benchmarks', {}),
            'total_duration': time.time() - self.overall_start_time if hasattr(self, 'overall_start_time') else 0
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(benchmark_dict, f, indent=2, default=str)
        
        print(f"Benchmark report saved to: {filepath}")
        print(f"JSON data saved to: {json_filepath}")
        
        return filepath
    def parse_multi_prompts(self, multi_prompts, num_sections):
        """
        Parse multi-line prompts and assign them to sections.
        Each line represents a prompt for a section.
        If fewer prompts than sections, the last prompt is repeated.
        """
        # Split by newline and filter empty lines
        prompts = [p.strip() for p in multi_prompts.split('\n') if p.strip()]
        
        if not prompts:
            raise ValueError("No prompts provided in multi_prompts")
        
        # Assign prompts to sections
        section_prompts = []
        for section in range(num_sections):
            if section < len(prompts):
                section_prompts.append(prompts[section])
            else:
                # Repeat the last prompt for remaining sections
                section_prompts.append(prompts[-1])
        
        print(f"Parsed {len(prompts)} unique prompts for {num_sections} sections")
        for i, prompt in enumerate(section_prompts):
            print(f"  Section {i}: {prompt[:50]}...")
        
        return section_prompts
    def encode_prompt_for_section(self, prompt, negative_prompt, text_encoder=None, device=None):
        """
        Encode a single prompt for a section using the WAN Video text encoder.
        Supports weighted prompts using (text:weight) syntax.
        """
        if device is None:
            device = self.device
        
        if text_encoder is None:
            raise ValueError("Text encoder is required for encoding prompts")
        
        # Extract the encoder model and dtype
        encoder = text_encoder["model"]
        dtype = text_encoder["dtype"]
        
        # Split positive prompts by '|' and process weights
        positive_prompts_raw = [p.strip() for p in prompt.split('|')]
        positive_prompts = []
        all_weights = []
        
        for p in positive_prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            positive_prompts.append(cleaned_prompt)
            all_weights.append(weights)
        
        # Move encoder to device
        encoder.model.to(device)
        
        try:
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype, enabled=True):
                # Encode positive and negative prompts
                context = encoder(positive_prompts, device)
                context_null = encoder([negative_prompt if negative_prompt else ""], device)
                
                # Apply weights to embeddings if any were extracted
                for i, weights in enumerate(all_weights):
                    if weights:  # Only apply if weights exist
                        for text, weight in weights.items():
                            log.info(f"Applying weight {weight} to prompt: {text}")
                            context[i] = context[i] * weight
        finally:
            # Always move encoder back to CPU to free VRAM
            offload_device = 'cpu'
            encoder.model.to(offload_device)
            mm.soft_empty_cache()
        
        # Create the embedding dictionary with all required fields for WAN Video
        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
        }
        
        return prompt_embeds_dict

    def parse_prompt_weights(self, prompt):
        """
        Parse prompt weights in the format (text:weight).
        Returns cleaned prompt and weight dictionary.
        """
        import re
        
        weights = {}
        cleaned_prompt = prompt
        
        # Pattern to find (text:weight) format
        pattern = r'\(([^:)]+):([0-9.]+)\)'
        matches = re.findall(pattern, prompt)
        
        for text, weight_str in matches:
            try:
                weight = float(weight_str)
                weights[text.strip()] = weight
                # Remove the weight notation from the prompt
                cleaned_prompt = cleaned_prompt.replace(f"({text}:{weight_str})", text)
            except ValueError:
                log.warning(f"Invalid weight value: {weight_str}")
        
        return cleaned_prompt.strip(), weights
    def process(self, model, vae, steps, cfg, shift, seed, scheduler,
            num_frames, width, height, force_offload, multi_prompts,
            encode_prompts=True, ref_images=None, input_frames=None, 
            input_mask=None, negative_prompt="", sigmas=None, 
            text_embeds_list=None, wan_t5_model=None, start_step=0, end_step=-1):
        """Main processing function for ComfyUI with multi-prompt support"""
        enable_benchmarking=True
        benchmark_output_dir="./benchmarks"
       

        
        if enable_benchmarking:
            self.overall_start_time = time.time()
            self.generation_params = {
                'num_frames': num_frames,
                'width': width,
                'height': height,
                'steps': steps,
                'cfg': cfg,
                'scheduler': scheduler,
                'seed': seed,
            }
            print("\n🔬 Benchmarking enabled - tracking performance metrics...")
        text_encoder= wan_t5_model
        device = mm.get_torch_device()
        self.device = device
        offload_device = mm.unet_offload_device()
        
        # Extract model components
        model_obj = model.model
        model_wrapper = model_obj.diffusion_model
        
        # Move models to device
        self.vae = vae.to(device).to(torch.float32)
        self.vae.dtype = torch.float32 
        model_wrapper.to(device)
        
        # Ensure dimensions are multiples of 16
        width = (width // 16) * 16
        height = (height // 16) * 16
        
        if True:
            # Multi-prompt mode - encode each prompt
            print("Multi-prompt mode activated")
            print("multi prompt", multi_prompts)
            # Calculate number of sections
            INITIAL_FRAMES = 81
            if num_frames <= INITIAL_FRAMES:
                num_sections = 1
            else:
                num_sections = math.ceil(num_frames / INITIAL_FRAMES)
            
            # Parse and encode prompts afor each section
            section_prompts = self.parse_multi_prompts(multi_prompts, num_sections)
            print('this are the section prompts',section_prompts )
            # Encode each prompt if we have a text encoder
            if text_encoder is not None:
                print("Encoding prompts for each section...")
                section_text_embeds = []
                for i, prompt in enumerate(section_prompts):
                    print(f"Encoding prompt {i+1}/{num_sections}: {prompt[:50]}...")
                    text_embed = self.encode_prompt_for_section(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        text_encoder=text_encoder,
                        device=device
                    )
                    section_text_embeds.append(text_embed)
            elif text_embeds_list:
                # Use pre-encoded embeddings if provided
                section_text_embeds = text_embeds_list
            else:
                # Fallback: use base embeddings for all sections
                print("Warning: No text encoder available, using base embeddings for all sections")
                section_text_embeds = [text_embeds] * num_sections
            
            latents = self._generate_with_framepack_multi(
                model_wrapper=model_wrapper,
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
                steps=steps,
                cfg=cfg,
                seed=seed,
                sigmas=sigmas,
                device=device,
                offload_device=offload_device,
                force_offload=force_offload
            )
        else:
            # Original single-prompt mode
            print("Single prompt mode")
            latents = self._generate_with_framepack(
                model_wrapper=model_wrapper,
                text_embeds=text_embeds,
                input_frames=input_frames,
                input_masks=input_mask,
                ref_images=ref_images,
                width=width,
                height=height,
                num_frames=num_frames,
                shift=shift,
                scheduler_name=scheduler,
                steps=steps,
                cfg=cfg,
                seed=seed,
                sigmas=sigmas,
                device=device,
                offload_device=offload_device,
                force_offload=force_offload
            )
            
        if enable_benchmarking:
            report = self.generate_benchmark_report()
            print("\n" + report)
            self.save_benchmark_report(report, benchmark_output_dir)
        else:
            report = "Benchmarking disabled"
        
        return ({"samples": latents.unsqueeze(0).cpu()}, )

    def _generate_with_framepack_multi(self, model_wrapper, section_text_embeds, 
                               section_prompts, input_frames, input_masks, 
                               ref_images, width, height, num_frames,
                               shift, scheduler_name, steps, cfg, seed, sigmas,
                               device, offload_device, force_offload):
        """Core FramePack generation algorithm with multi-prompt support"""
        vae_dtype = torch.float32
        all_generated_latents = []  
        accumulated_latents = [] 
        total_output_frames = 0

        LATENT_WINDOW = 41  
        GENERATION_FRAMES = 30
        CONTEXT_FRAMES = 11
        INITIAL_FRAMES = 81
        enable_benchmarking = True
        
        if num_frames <= INITIAL_FRAMES:
            num_sections = 1
        else:
            num_sections = math.ceil(num_frames / INITIAL_FRAMES)
        
        for section in range(num_sections):
            print(f"\n[Section {section+1}/{num_sections}]")
            print(f"Using prompt: {section_prompts[section][:100]}...")
            
            # Get text embeddings for this section
            text_embeds = section_text_embeds[section]
            
            # PHASE 1: ENCODING
            if enable_benchmarking:
                self.benchmark_section(section, 'encoding')
                
            if section == 0:
                input_frames = torch.zeros(1, 3, INITIAL_FRAMES, height, width, 
                                        device=device, dtype=vae_dtype)
                input_masks = torch.ones_like(input_frames, device=device, dtype=vae_dtype)
                
                # Normalize to [-1, 1] range for VAE
                input_frames = [(f * 2 - 1) for f in input_frames]
                
                if ref_images is not None:
                    # [ref_images processing code - same as before]
                    if ref_images.shape[0] > 1:
                        ref_images = torch.cat([ref_images[i] for i in range(ref_images.shape[0])], 
                                            dim=1).unsqueeze(0)
                
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
                                            "lanczos", "center").movedim(1, -1)
                    ref_images = ref_images.to(self.vae.dtype).to(self.device).unsqueeze(0)
                    ref_images = ref_images.permute(0, 4, 1, 2, 3).unsqueeze(0)
                    ref_images = ref_images * 2 - 1
                    
                target_shape = (
                    16,
                    (INITIAL_FRAMES - 1) // VAE_STRIDE[0] + 1,
                    height // VAE_STRIDE[1],
                    width // VAE_STRIDE[2]
                )
            else:
                # Build context from previous sections
                context_latent = self.build_hierarchical_context_latent(accumulated_latents, section)
                hierarchical_frames = self.pick_context_v2(context_latent, section)
                
                input_frames = self.decode_latent([hierarchical_frames], None)
                input_frames[0] = input_frames[0].expand(3, -1, -1, -1)
                
                input_masks = self.create_temporal_blend_mask_v2(input_frames[0].shape, section)
                ref_images = None
                num_frames = input_frames[0].shape[1]
                
                target_shape = (
                    16,
                    (num_frames - 1) // VAE_STRIDE[0] + 1,
                    height // VAE_STRIDE[1],
                    width // VAE_STRIDE[2]
                )
            
            # Encode inputs to latent space
            z0 = self.vace_encode_frames(input_frames, ref_images=ref_images, 
                                        masks=input_masks, tiled_vae=False)
            m0 = self.vace_encode_masks(input_masks, ref_images=ref_images)
            z = self.vace_latent(z0, m0)
            
            if enable_benchmarking:
                self.benchmark_section(section, 'encoding')  # End encoding
            
            # PHASE 2: DENOISING
            if enable_benchmarking:
                self.benchmark_section(section, 'denoising')  # Start denoising
            
            # Setup scheduler
            sample_scheduler = self._setup_scheduler(scheduler_name, steps, shift, device, sigmas)
            timesteps = sample_scheduler.timesteps
            
            # Initialize noise
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed if seed != -1 else torch.randint(0, 2**32, (1,)).item())
            
            has_ref = ref_images is not None
            noise = torch.randn(
                target_shape[0],
                target_shape[1] + (1 if has_ref else 0),
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device="cpu",
                generator=generator
            )
            
            latent = noise.to(device)
            
            # Setup model parameters
            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1])
            freqs = self._setup_rope_embeddings(model_wrapper, latent.shape[1])
            num_steps = len(timesteps)

            vace_data = [{
                "context": z,
                "scale": [1.0] * num_steps,
                "start": 0.0,
                "end": 1.0,
                "seq_len": seq_len
            }]
            
            # Ensure cfg is a list
            if not isinstance(cfg, list):
                cfg = [cfg] * (steps + 1)
            
            # Setup progress bar
            pbar = ProgressBar(steps)
            
            # Clear memory before generation
            mm.soft_empty_cache()
            gc.collect()
            
            # Initialize cache state
            self.cache_state = [None, None]
            
            # Main denoising loop
            for idx, t in enumerate(timesteps):
                timestep = torch.tensor([t]).to(device)
                
                # Get noise prediction
                noise_pred = self._predict_with_cfg(
                    latent=latent,
                    cfg_scale=cfg[idx],
                    text_embeds=text_embeds,
                    timestep=timestep,
                    idx=idx,
                    model_wrapper=model_wrapper,
                    vace_data=vace_data,
                    seq_len=seq_len,
                    freqs=freqs,
                    device=device
                )
                
                # Scheduler step
                step_args = {"generator": generator}
                if isinstance(sample_scheduler, (DEISMultistepScheduler, FlowMatchScheduler)):
                    step_args.pop("generator", None)
                
                latent = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    **step_args
                )[0].squeeze(0)
                
                pbar.update(1)
                
                # Memory management - less frequent
                if force_offload and idx % 10 == 0:
                    mm.soft_empty_cache()
            
            if enable_benchmarking:
                self.benchmark_section(section, 'denoising')  # End denoising
            
            # PHASE 3: ACCUMULATION
            if enable_benchmarking:
                self.benchmark_section(section, 'accumulation')  # Start accumulation
            
            # Handle accumulation based on section
            if section == 0:
                if ref_images is not None:
                    latent_without_ref = latent[:, 1:, :, :]
                else:
                    latent_without_ref = latent
                
                accumulated_latents.append(latent_without_ref)
                all_generated_latents.append(latent_without_ref)
            else:
                # Remove oldest section if we have too many
                if section > 2:
                    accumulated_latents.pop(0)
                
                # Add new frames to accumulation
                new = latent[:, -GENERATION_FRAMES:, :, :]
                accumulated_latents.append(new)
                
                # Add to final output (skip overlap frames)
                new_content = latent[:, -GENERATION_FRAMES:, :, :]
                new_content = new_content[:, CONTEXT_FRAMES:, :, :]
                all_generated_latents.append(new_content)
                
                frames_added = new_content.shape[1]
                total_output_frames += frames_added
                print(f"Added {frames_added} frames (total: {total_output_frames})")
            
            if enable_benchmarking:
                self.benchmark_section(section, 'accumulation')  # End accumulation
            
            # Clear cache after section
            if 'noise_pred' in locals(): 
                del latent, noise_pred
            mm.soft_empty_cache()
            gc.collect()
        
        # Move model to offload device if requested
        if force_offload:
            model_wrapper.to(offload_device)
            mm.soft_empty_cache()
            gc.collect()
        
        final_latent = torch.cat(all_generated_latents, dim=1)
        return final_latent.cpu()

    def _predict_with_cfg(self, latent, cfg_scale, text_embeds, timestep, idx,
                         model_wrapper, vace_data, seq_len, freqs, device):
        """Classifier-free guidance prediction"""
        
        dtype = torch.float32
        latent = latent.to(dtype)
        
        with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype):
            # Prepare base parameters
            base_params = {
                'seq_len': seq_len,
                'device': device,
                'freqs': freqs,
                't': timestep,
                'current_step': idx,
                "nag_params": text_embeds.get("nag_params", {}),
                "nag_context": text_embeds.get("nag_prompt_embeds", None),
                "ref_target_masks": None
            }
            
            current_step_percentage = idx / 30  # Approximate total steps
            
            # Conditional prediction
            noise_pred_cond, cache_state_cond = model_wrapper(
                [latent],
                context=text_embeds["prompt_embeds"],
                y=None,
                clip_fea=None,
                is_uncond=False,
                current_step_percentage=current_step_percentage,
                pred_id=self.cache_state[0] if self.cache_state else None,
                vace_data=vace_data,
                attn_cond=None,
                **base_params
            )
            noise_pred_cond = noise_pred_cond[0]
            
            # If cfg_scale is 1.0, skip unconditional
            if math.isclose(cfg_scale, 1.0):
                self.cache_state = [cache_state_cond, None]
                return noise_pred_cond
            
            # Unconditional prediction
            noise_pred_uncond, cache_state_uncond = model_wrapper(
                [latent],
                context=text_embeds["negative_prompt_embeds"],
                y=None,
                clip_fea=None,
                is_uncond=True,
                current_step_percentage=current_step_percentage,
                pred_id=self.cache_state[1] if self.cache_state else None,
                vace_data=vace_data,
                attn_cond=None,
                **base_params
            )
            noise_pred_uncond = noise_pred_uncond[0]
            
            # Apply CFG
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Update cache state
            self.cache_state = [cache_state_cond, cache_state_uncond]
            
            return noise_pred

    def _setup_scheduler(self, scheduler_name, steps, shift, device, sigmas=None):
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

    def _setup_rope_embeddings(self, model_wrapper, latent_video_length):
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
    def vace_encode_frames(self, frames, ref_images, masks=None,tiled_vae=False):
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


    def decode_latents(self, latents, device):
        """Decode latents back to video"""
        
        with torch.no_grad():
            if hasattr(self.vae, 'decode'):
                # Use VAE decoder
                video = self.vae.decode(latents, device)
                if hasattr(video, 'sample'):
                    video = video.sample
            else:
                # Fallback: simple upsampling
                video = F.interpolate(latents.unsqueeze(0), scale_factor=8, mode='trilinear')
                video = video.squeeze(0)
        
        # Convert from [-1, 1] to [0, 1]
        video = (video + 1) / 2
        video = video.clamp(0, 1)
        
        # Rearrange to ComfyUI format [T, H, W, C]
        if len(video.shape) == 4:  # [C, T, H, W]
            video = video.permute(1, 2, 3, 0)
        elif len(video.shape) == 5:  # [B, C, T, H, W]
            video = video.squeeze(0).permute(1, 2, 3, 0)
        
        return video
    
    def decode_latent(self, zs, ref_images=None, vae=None):
        vae = self.vae if vae is None else vae

        # No need to check ref_images length or trim anymore
        return vae.decode(zs, device=self.device)

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
        
        mask = torch.zeros(3, decoded_frames, H, W, device=self.device)
        
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

# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": WanVACEVideoFramepackSampler2
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": "WanVACE FramePack Sampler 2"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']