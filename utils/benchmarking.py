"""
Benchmarking utilities for FramePack video generation.
Contains BenchmarkManager for per-run tracking and BenchmarkAnalyzer for cross-run comparison.
"""

import torch
import time
import json
import os
from datetime import datetime

import psutil
from pynvml import (
    nvmlInit, nvmlShutdown, nvmlDeviceGetCount,
    nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates
)


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

    def log_metric(self, section_id, metric_name, value):
        """Log a custom quality metric for a section"""
        if section_id not in self.section_benchmarks:
            self.section_benchmarks[section_id] = {}
        self.section_benchmarks[section_id][metric_name] = value
    
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


class BenchmarkAnalyzer:
    """Consolidates results from multiple runs into a comparison report."""
    
    def __init__(self, output_dir="./benchmarks"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_run_data(self, method_name, benchmark_manager):
        """Save a single run's data to a JSON file"""
        data = {
            "method": method_name,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "params": benchmark_manager.generation_params,
            "sections": benchmark_manager.section_benchmarks
        }
        
        filename = os.path.join(self.output_dir, f"run_{method_name}.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved benchmark data for method '{method_name}' to {filename}")

    def generate_comparison_report(self):
        """Build a Markdown comparison report from all JSON files in the directory"""
        files = [f for f in os.listdir(self.output_dir) if f.startswith("run_") and f.endswith(".json")]
        if not files:
            return "No benchmark data found to compare."
            
        all_data = []
        for file in files:
            with open(os.path.join(self.output_dir, file), 'r') as f:
                all_data.append(json.load(f))
                
        report = ["# FramePack Context Method Comparison Report\n"]
        report.append("| Method | Avg. SSIM (Motion) | Identity Drift | VRAM (Max GB) | Speed (FPS) |")
        report.append("| :--- | :---: | :---: | :---: | :---: |")
        
        for run in all_data:
            method = run["method"]
            sections = run["sections"]
            
            # Aggregate stats
            ssims = [v["boundary_ssim"] for v in sections.values() if "boundary_ssim" in v]
            drifts = [v["identity_drift"] for v in sections.values() if "identity_drift" in v]
            vrams = [v["denoising_memory_end"]["gpu_memory_allocated_gb"] for v in sections.values() if "denoising_memory_end" in v]
            durations = [v["denoising_duration"] for v in sections.values() if "denoising_duration" in v]
            
            avg_ssim = sum(ssims)/len(ssims) if ssims else 0
            avg_drift = sum(drifts)/len(drifts) if drifts else 0
            max_vram = max(vrams) if vrams else 0
            
            total_frames = run["params"].get("num_frames", 30)
            total_duration = sum(durations) if durations else 1
            fps = total_frames / total_duration
            
            report.append(f"| {method} | {avg_ssim:.4f} | {avg_drift:.4f} | {max_vram:.2f} | {fps:.2f} |")
            
        report_text = "\n".join(report)
        
        with open(os.path.join(self.output_dir, "comparison_report.md"), 'w') as f:
            f.write(report_text)
            
        return report_text
