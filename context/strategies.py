"""
Context selection strategies for FramePack video generation.
Contains SparseSelector, MoCRouter, and FramePackCompressor.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


class FramePackCompressor:
    """
    Handles hierarchical compression of latent history in latent space.
    Recent history stays high-res; deep history is pooled.
    """
    def __init__(self, 
                 lambda_compression: float = 2.0,
                 max_history_frames: int = 120):
        
        self.lambda_compression = lambda_compression
        self.max_history_frames = max_history_frames
        
        # Kernel sizes for different levels of compression
        self.base_kernels = [
            (1, 1, 1),   # Level 0: 1x
            (1, 2, 2),   # Level 1: 4x
            (2, 2, 2),   # Level 2: 8x
            (2, 4, 4),   # Level 3: 32x
            (4, 8, 8),   # Level 4: 256x
        ]

    def _get_kernel_for_section(self, section_age: int) -> Tuple[int, int, int]:
        """Calculates kernel size based on section age and lambda"""
        if section_age == 0:
            return (1, 1, 1)
        
        # Exponential growth of compression: lambda^(age)
        target_rate = self.lambda_compression ** section_age
        
        # Find best kernel from our presets
        best_kernel = self.base_kernels[0]
        for kernel in self.base_kernels:
            rate = kernel[0] * kernel[1] * kernel[2]
            if rate <= target_rate:
                best_kernel = kernel
            else:
                break
        return best_kernel

    def compress_latent(self, latent: torch.Tensor, kernel: Tuple[int, int, int]) -> torch.Tensor:
        """Applies avg_pool3d to compress latent dimensions"""
        if kernel == (1, 1, 1):
            return latent
        
        # latent is [C, T, H, W]
        # avg_pool3d expects [N, C, T, H, W]
        with torch.no_grad():
            compressed = F.avg_pool3d(
                latent.unsqueeze(0),
                kernel_size=kernel,
                stride=kernel
            ).squeeze(0)
        return compressed

    def prepare_context(self, accumulated_latents: List[torch.Tensor], section_id: int) -> torch.Tensor:
        """
        Builds a single context tensor from the list of historical latent sections.
        Sections are compressed according to their age.
        """
        if not accumulated_latents:
            return None
            
        # Reverse history so newest is index 0
        history = list(reversed(accumulated_latents))
        
        processed_sections = []
        for age, section in enumerate(history):
            kernel = self._get_kernel_for_section(age)
            compressed = self.compress_latent(section, kernel)
            processed_sections.append(compressed)
            
            # Limit history to prevent runaway sequence length
            if len(processed_sections) >= self.max_history_frames // 10:
                break
        
        # Concatenate into one temporal block
        # Note: Since they have different H, W after pooling, we pad to match the largest
        max_h = max(s.shape[2] for s in processed_sections)
        max_w = max(s.shape[3] for s in processed_sections)
        
        padded = []
        for s in processed_sections:
            if s.shape[2] < max_h or s.shape[3] < max_w:
                pad_h = max_h - s.shape[2]
                pad_w = max_w - s.shape[3]
                s = F.pad(s, (0, pad_w, 0, pad_h))
            padded.append(s)
            
        return torch.cat(padded, dim=1)


class SparseSelector:
    """
    Selects frames with exponential backoff to preserve long-term identity.
    Example: [Last 5 frames, then T-10, T-30, T-60, Frame 0]
    """
    @staticmethod
    def pick_sparse_context(accumulated_latents: List[torch.Tensor], num_frames: int = 30) -> torch.Tensor:
        if not accumulated_latents:
            return None
            
        all_frames = torch.cat(accumulated_latents, dim=1)
        total_t = all_frames.shape[1]
        
        if total_t <= num_frames:
            return all_frames
            
        # Select indices
        indices = []
        
        # 1. Always include the very first frame (Identity Anchor)
        indices.append(0)
        
        # 2. Always include the last N/2 frames (Continuity)
        recent_count = num_frames // 2
        indices.extend(range(total_t - recent_count, total_t))
        
        # 3. Fill the rest with exponential spacing from the past
        remaining = num_frames - len(indices)
        if remaining > 0:
            # Calculate points between Frame 1 and (Total - Recent)
            search_end = total_t - recent_count - 1
            if search_end > 1:
                # Use log space for selection
                log_points = np.linspace(np.log(1), np.log(search_end), remaining)
                sparse_indices = np.exp(log_points).astype(int)
                indices.extend(sparse_indices.tolist())
        
        # Unique and sorted
        final_indices = sorted(list(set(indices)))
        
        # If we have too many (due to overlaps), take the most important ones
        if len(final_indices) > num_frames:
            final_indices = final_indices[-num_frames:]
            
        return all_frames[:, final_indices, :, :]


class MoCRouter:
    """
    Mixture of Contexts Router (Outer Loop).
    Retrieves historical chunks based on semantic similarity to the current prompt.
    """
    @staticmethod
    def retrieve_context(accumulated_latents: List[torch.Tensor], 
                         current_prompt_embeds: torch.Tensor,
                         top_k: int = 3) -> torch.Tensor:
        """
        current_prompt_embeds: [1, L, D]
        returns: Concatenated top-k most similar latent chunks.
        """
        if not accumulated_latents:
            return None
            
        if len(accumulated_latents) <= top_k:
            return torch.cat(accumulated_latents, dim=1)
            
        # 1. Prepare Query: Mean pool the prompt embeddings
        # current_prompt_embeds is usually a list or dict in this node
        if isinstance(current_prompt_embeds, dict):
            query = current_prompt_embeds["prompt_embeds"]
        else:
            query = current_prompt_embeds
            
        # If it's a list, take the first one
        if isinstance(query, list):
            query = query[0]
            
        # query shape: [1, seq_len, dim] -> [dim]
        query_vec = query.mean(dim=1).flatten()
        
        # 2. Prepare Keys: Mean pool each historical chunk
        chunk_scores = []
        for i, chunk in enumerate(accumulated_latents):
            # chunk shape: [C, T, H, W] -> [C]
            key_vec = chunk.mean(dim=(1, 2, 3)).flatten()
            
            # 3. Calculate Cosine Similarity
            # Note: Wan Video latents (16 channels) and T5 embeds (4096 dim) 
            # don't match directly. We use high-level spatial correlations.
            
            # ZERO-SHOT TRANSLATION:
            # Inner Loop: Current Query (Q) vs Historical Keys (K)
            # Outer Loop (Ours): Last Frame Feature (Query Ref) vs Historical Chunk Means (Keys)
            
            # Query Ref: Mean-pool the very last frame of the most recent chunk
            # accumulated_latents[-1] is [C, T, H, W] -> [C, -1, :, :] -> [C]
            ref_frame_vec = accumulated_latents[-1][:, -1, :, :].mean(dim=(1, 2)).flatten()
            
            similarity = F.cosine_similarity(ref_frame_vec.unsqueeze(0), key_vec.unsqueeze(0))
            chunk_scores.append((i, similarity.item()))
            
        # 4. Select Top-K
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([x[0] for x in chunk_scores[:top_k]])
        
        # 5. Always include the most recent chunk for continuity
        recent_idx = len(accumulated_latents) - 1
        if recent_idx not in selected_indices:
            selected_indices[-1] = recent_idx # Replace least similar with most recent
            selected_indices.sort()
            
        selected_chunks = [accumulated_latents[i] for i in selected_indices]
        return torch.cat(selected_chunks, dim=1)
