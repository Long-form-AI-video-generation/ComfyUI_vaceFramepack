"""
FramePack Helper Functions
Contains utility functions for video generation, encoding, context building, and benchmarking.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math
import time
import gc
import json
import os
from datetime import datetime
from tqdm import tqdm
import psutil

from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)

from comfy import model_management as mm
from comfy.utils import common_upscale

from .wanvideo.modules.model import rope_params
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
from .wanvideo.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .wanvideo.utils.basic_flowmatch import FlowMatchScheduler
from .wanvideo.utils.scheduling_flow_match_lcm import FlowMatchLCMScheduler


# Constants
VAE_STRIDE = (4, 8, 8)
PATCH_SIZE = (1, 2, 2)


class BenchmarkManager:
    """Manages benchmarking and performance tracking"""
    
    def __init__(self):
        self.section_benchmarks = {}
        self.overall_start_time = None
        self.generation_params = {}
    
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
                    handle = nvmlDeviceGetHandleByIndex(0)
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
    
    def generate_report(self, section_prompts=None):
        """Generate a comprehensive benchmark report"""
        report = []
        report.append("=" * 80)
        report.append("FRAMEPACK VIDEO GENERATION BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if self.overall_start_time:
            total_duration = time.time() - self.overall_start_time
            report.append(f"Total Processing Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        
        if self.generation_params:
            report.append("\nGeneration Parameters:")
            for key, value in self.generation_params.items():
                report.append(f"  {key}: {value}")
        
        report.append("\n" + "-" * 80)
        report.append("SECTION-BY-SECTION BREAKDOWN")
        report.append("-" * 80)
        
        total_encoding_time = 0
        total_denoising_time = 0
        total_accumulation_time = 0
        
        for section_id in sorted(self.section_benchmarks.keys()):
            section_data = self.section_benchmarks[section_id]
            report.append(f"\n[Section {section_id + 1}]")
            
            # Prompt info
            if section_prompts and section_id < len(section_prompts):
                report.append(f"Prompt: {section_prompts[section_id][:50]}...")
            
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
    
    def save_report(self, report, output_dir="./benchmarks"):
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
            'generation_params': self.generation_params,
            'section_benchmarks': self.section_benchmarks,
            'total_duration': time.time() - self.overall_start_time if self.overall_start_time else 0
        }
        
        with open(json_filepath, 'w') as f:
            json.dump(benchmark_dict, f, indent=2, default=str)
        
        print(f"Benchmark report saved to: {filepath}")
        print(f"JSON data saved to: {json_filepath}")
        
        return filepath


class PromptHandler:
    """Handles prompt parsing and encoding"""
    
    @staticmethod
    def parse_multi_prompts(multi_prompts, num_sections):
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
    
    @staticmethod
    def parse_prompt_weights(prompt):
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
                print(f"Invalid weight value: {weight_str}")
        
        return cleaned_prompt.strip(), weights
    
    @staticmethod
    def encode_prompt_for_section(prompt, negative_prompt, text_encoder, device):
        """
        Encode a single prompt for a section using the WAN Video text encoder.
        Supports weighted prompts using (text:weight) syntax.
        """
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
            cleaned_prompt, weights = PromptHandler.parse_prompt_weights(p)
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
                            print(f"Applying weight {weight} to prompt: {text}")
                            context[i] = context[i] * weight
        finally:
            # Always move encoder back to CPU to free VRAM
            encoder.model.to('cpu')
            mm.soft_empty_cache()
        
        # Create the embedding dictionary with all required fields for WAN Video
        prompt_embeds_dict = {
            "prompt_embeds": context,
            "negative_prompt_embeds": context_null,
        }
        
        return prompt_embeds_dict


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
                assert all([x.shape[1] == 1 for x in ref_latent])
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
                length = len(refs)
                mask_pad = torch.zeros_like(mask[:, :length, :, :])
                mask = torch.cat((mask_pad, mask), dim=1)
            result_masks.append(mask)
        
        return result_masks
    
    def combine_latent(self, z, m):
        """Combine latents and masks"""
        return [torch.cat([zz, mm], dim=0) for zz, mm in zip(z, m)]

    def decode_latent(self, zs, ref_images=None):
        """Decode latents back to frames"""
        return self.vae.decode(zs, device=self.device)


class ContextBuilder:
    """Handles hierarchical context building and frame selection"""
    
    @staticmethod
    def build_hierarchical_context(accumulated_latents, section_id):
        """Build hierarchical context from accumulated latents"""
        if not accumulated_latents:
            raise ValueError("No accumulated latents available")

        all_prev = torch.cat(accumulated_latents, dim=1)
        total_frames = all_prev.shape[1]

        print(f"Building context from {total_frames} accumulated frames")

        return all_prev
    
    @staticmethod
    def pick_context(frames, section_id, initial=False):
        """
        Enhanced hierarchical context selection with constant 41-frame output.
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

        # Long-term context
        if T >= 40:
            step = max(4, T // 20)
            long_indices = []
            for i in range(LONG_FRAMES):
                idx = min(i * step, T - 15)
                long_indices.append(idx)
            selected_indices.extend(long_indices)
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
        mid_indices = [
            min(mid_start, T - 1),
            min(mid_start + 2, T - 1)
        ]
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


class MaskGenerator:
    """Handles mask generation for temporal blending"""
    
    @staticmethod
    def create_temporal_blend_mask(frame_shape, section_id, device, initial=False):
        """Enhanced mask creation that handles decoded frame dimensions"""
        C, T, H, W = frame_shape
        
        # Constants
        LATENT_FRAMES = 41
        decoded_frames = T
        expansion_ratio = decoded_frames / LATENT_FRAMES
        
        mask = torch.zeros(3, decoded_frames, H, W, device=device)
        
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
        
        # Apply mask values
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
    
    @staticmethod
    def create_spatial_variation(H, W, device):
        """Create spatial variation mask for natural blending"""
        y_coords = torch.linspace(-1, 1, H, device=device)
        x_coords = torch.linspace(-1, 1, W, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        distance = torch.sqrt(x_grid**2 + y_grid**2) / 1.414
        variation = 1.0 - 0.3 * torch.exp(-3 * distance**2)

        return variation


class FrequencyProcessor:
    """Handles frequency domain operations"""
    
    @staticmethod
    def separate_appearance_and_motion(frames):
        """Use frequency domain to separate appearance from motion"""
        C, T, H, W = frames.shape

        # FFT
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
            low_pass_mask = low_pass_mask[:fft_h, :fft_w]

        while low_pass_mask.dim() < fft_frames.dim():
            low_pass_mask = low_pass_mask.unsqueeze(0)

        appearance_fft = fft_frames * low_pass_mask
        motion_fft = fft_frames * (1 - low_pass_mask)

        appearance = torch.fft.irfft2(appearance_fft, s=(H, W))
        motion = torch.fft.irfft2(motion_fft, s=(H, W))

        return appearance, motion


class ReferenceImageProcessor:
    """Handles reference image processing"""
    
    @staticmethod
    def process_reference_images(ref_images, width, height, device, dtype):
        """Process reference images for generation"""
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
        ref_images = ref_images.to(dtype).to(device).unsqueeze(0)
        ref_images = ref_images.permute(0, 4, 1, 2, 3).unsqueeze(0)
        ref_images = ref_images * 2 - 1
        
        return ref_images