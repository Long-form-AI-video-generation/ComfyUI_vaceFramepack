"""
Context building utilities for FramePack video generation.
Contains hierarchical context building, mask generation, and frequency processing.
"""

import torch


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
        LONG_FRAMES = 14
        MID_FRAMES = 8
        RECENT_FRAMES = 3
        OVERLAP_FRAMES = 5
        GEN_FRAMES = 30
        TOTAL_FRAMES = 60

        C, T, H, W = frames.shape

        if initial and T == TOTAL_FRAMES:
            return frames

        if initial and T < TOTAL_FRAMES:
            padding_needed = TOTAL_FRAMES - T
            padding = torch.zeros((C, padding_needed, H, W), device=frames.device)
            return torch.cat([frames, padding], dim=1)

        # SIMPLIFIED STRATEGY: Use the last 11 frames contiguously
        CONTEXT_FRAMES = LONG_FRAMES + MID_FRAMES + RECENT_FRAMES + OVERLAP_FRAMES # 11
        
        # Ensure we have enough frames
        if T < CONTEXT_FRAMES:
             # Fallback if not enough frames (unlikely after section 0)
             context_frames = frames
             padding = torch.zeros((C, CONTEXT_FRAMES - T, H, W), device=frames.device)
             context_frames = torch.cat([padding, context_frames], dim=1)
        else:
             context_frames = frames[:, -CONTEXT_FRAMES:, :, :]

        # Return ONLY the context frames, do not pad with placeholder yet
        # We will handle padding in pixel space after decoding
        final_frames = context_frames

        if section_id % 5 == 0 or True: # Always print for now
            print(f"\nContext selection debug (section {section_id}):")
            print(f"  Input frames: {T}")
            print(f"  Strategy: Contiguous last {CONTEXT_FRAMES} frames")
            print(f"  Output shape: {final_frames.shape}")

        return final_frames


class MaskGenerator:
    """Handles mask generation for temporal blending"""
    
    @staticmethod
    def create_temporal_blend_mask(frame_shape, section_id, device, initial=False):
        """Enhanced mask creation that handles decoded frame dimensions"""
        C, T, H, W = frame_shape
        
        # Constants
        LATENT_FRAMES = 60
        decoded_frames = T
        expansion_ratio = decoded_frames / LATENT_FRAMES
        
        mask = torch.zeros(3, decoded_frames, H, W, device=device)
        
        # Scale all frame counts by the expansion ratio
        LONG_FRAMES = int(14 * expansion_ratio)
        MID_FRAMES = int(8 * expansion_ratio)
        RECENT_FRAMES = int(3 * expansion_ratio)
        OVERLAP_FRAMES = int(5 * expansion_ratio)
        GEN_FRAMES = decoded_frames - (LONG_FRAMES + MID_FRAMES + RECENT_FRAMES + OVERLAP_FRAMES)
        
        print(f"\nMask generation debug (section {section_id}):")
        print(f"  Decoded frames: {decoded_frames}")
        print(f"  Expansion ratio: {expansion_ratio}")
        print(f"  Segments: L={LONG_FRAMES}, M={MID_FRAMES}, R={RECENT_FRAMES}, O={OVERLAP_FRAMES}, Gen={GEN_FRAMES}")
        
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
