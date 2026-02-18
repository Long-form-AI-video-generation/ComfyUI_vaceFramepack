"""
Video quality and consistency metrics for FramePack video generation.
Provides SSIM boundary measurement, embedding drift tracking,
temporal coherence analysis, and semantic alignment scoring.
"""

import torch
import torch.nn.functional as F
import math


class VideoMetrics:
    """Calculates quality and consistency metrics for video generation."""

    # ------------------------------------------------------------------ #
    #                       SSIM (Boundary Quality)                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
        """Create a 1-D Gaussian kernel for windowed SSIM."""
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    @staticmethod
    def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
        """Create a 2-D Gaussian kernel suitable for depthwise conv2d."""
        k1d = VideoMetrics._gaussian_kernel_1d(size, sigma, device)
        k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)           # [size, size]
        k2d = k2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)  # [C, 1, size, size]
        return k2d

    @staticmethod
    def _ensure_4d(frame: torch.Tensor) -> torch.Tensor:
        """Squeeze/reshape an arbitrary frame tensor down to [1, C, H, W]."""
        while frame.dim() > 4:
            # Remove singleton dims first
            for d in range(frame.dim()):
                if frame.shape[d] == 1:
                    frame = frame.squeeze(d)
                    break
            else:
                # No singletons left — collapse leading dims
                frame = frame.reshape(-1, *frame.shape[-2:])
                break
        if frame.dim() == 2:        # [H, W]
            frame = frame.unsqueeze(0).unsqueeze(0)
        elif frame.dim() == 3:      # [C, H, W]
            frame = frame.unsqueeze(0)
        return frame  # [1, C, H, W]

    @staticmethod
    def calculate_ssim_boundary(
        frame_prev: torch.Tensor,
        frame_next: torch.Tensor,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 2.0,   # latents are roughly in [-1, 1]
        size_average: bool = True,
    ) -> float:
        """
        Windowed SSIM between two frames using a Gaussian kernel.

        Parameters
        ----------
        frame_prev, frame_next : Tensor
            Frames of any shape that can be reduced to [1, C, H, W].
        window_size : int
            Side length of the Gaussian window (default 11).
        sigma : float
            Standard deviation of the Gaussian window.
        data_range : float
            Dynamic range of the input (2.0 for data in [-1, 1]).
        size_average : bool
            If True return the scalar mean; else return the full SSIM map.

        Returns
        -------
        float
            SSIM value in [−1, 1] (higher is better).
        """
        x = VideoMetrics._ensure_4d(frame_prev)
        y = VideoMetrics._ensure_4d(frame_next)

        # Match spatial size
        if x.shape[-2:] != y.shape[-2:]:
            y = F.interpolate(y.float(), size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Match channel count (take the minimum)
        c = min(x.shape[1], y.shape[1])
        x, y = x[:, :c].float(), y[:, :c].float()

        # Create Gaussian kernel
        kernel = VideoMetrics._gaussian_kernel_2d(window_size, sigma, c, x.device)
        pad = window_size // 2

        # Luminance / contrast constants (Wang 2004)
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        mu_x  = F.conv2d(x, kernel, padding=pad, groups=c)
        mu_y  = F.conv2d(y, kernel, padding=pad, groups=c)

        mu_x_sq  = mu_x ** 2
        mu_y_sq  = mu_y ** 2
        mu_xy    = mu_x * mu_y

        sigma_x_sq  = F.conv2d(x * x, kernel, padding=pad, groups=c) - mu_x_sq
        sigma_y_sq  = F.conv2d(y * y, kernel, padding=pad, groups=c) - mu_y_sq
        sigma_xy    = F.conv2d(x * y, kernel, padding=pad, groups=c) - mu_xy

        # Clamp to avoid negative variance from floating-point errors
        sigma_x_sq = sigma_x_sq.clamp(min=0)
        sigma_y_sq = sigma_y_sq.clamp(min=0)

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
                   ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))

        if size_average:
            return float(ssim_map.mean().item())
        return ssim_map

    # ------------------------------------------------------------------ #
    #                     Embedding Drift (Identity)                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_embedding_drift(embed1: torch.Tensor, embed2: torch.Tensor) -> float:
        """
        Cosine distance between two embeddings.
        Returns 0.0 when identical, 2.0 when opposite.
        """
        sim = F.cosine_similarity(embed1.view(1, -1).float(),
                                  embed2.view(1, -1).float())
        return float(1.0 - sim.item())

    # ------------------------------------------------------------------ #
    #                    Temporal Coherence (Multi-frame)                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_temporal_coherence(
        latent_sequence: torch.Tensor,
        window: int = 3,
    ) -> dict:
        """
        Measure temporal smoothness across a latent video.

        Parameters
        ----------
        latent_sequence : Tensor  [C, T, H, W]
            A sequence of latent frames.
        window : int
            Number of consecutive pairs to average over.

        Returns
        -------
        dict with keys:
            mean_ssim      – average pairwise SSIM across the window
            max_drift      – maximum frame-to-frame change (L2)
            smoothness     – 1 / (1 + variance of frame-to-frame diffs)
        """
        C, T, H, W = latent_sequence.shape
        if T < 2:
            return {"mean_ssim": 1.0, "max_drift": 0.0, "smoothness": 1.0}

        pairs = min(window, T - 1)
        ssims, drifts = [], []
        for i in range(T - pairs, T - 1):
            f1 = latent_sequence[:, i, :, :]      # [C, H, W]
            f2 = latent_sequence[:, i + 1, :, :]
            ssims.append(VideoMetrics.calculate_ssim_boundary(f1, f2))
            drifts.append(float((f2 - f1).norm().item()))

        diff_norms = torch.tensor(drifts)
        smoothness = float(1.0 / (1.0 + diff_norms.var().item()))

        return {
            "mean_ssim": sum(ssims) / len(ssims),
            "max_drift": max(drifts),
            "smoothness": smoothness,
        }

    # ------------------------------------------------------------------ #
    #                   Semantic Alignment (Latent ↔ Prompt)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def calculate_semantic_alignment(
        latent: torch.Tensor,
        prompt_embed: torch.Tensor,
        projection_dim: int = 128,
        seed: int = 42,
    ) -> float:
        """
        Measure how well a generated latent aligns with the conditioning prompt.

        The latent space (C=16 channels) and the text embedding space (D=4096)
        live in different dimensionalities. We bridge them with a **deterministic
        random projection** (Johnson-Lindenstrauss) that maps both vectors into
        a shared low-dimensional space, then compute cosine similarity.

        This is training-free and reproducible (fixed seed).

        Parameters
        ----------
        latent : Tensor  [C, T, H, W]   — e.g. [16, 21, 60, 104]
        prompt_embed : Tensor  [1, L, D] — e.g. [1, 512, 4096]
        projection_dim : int
            Target dimensionality for the random projection (default 128).
        seed : int
            Fixed seed for the projection matrices so results are reproducible.

        Returns
        -------
        float
            Cosine similarity in [−1, 1].  Higher ≈ better alignment.
        """
        device = latent.device
        dtype = torch.float32

        # ---- 1. Pool both modalities into flat vectors ---- #
        # Latent: [C, T, H, W] → spatial-temporal statistics [C * 4]
        # We use (mean, std, min, max) over T×H×W to capture the distribution.
        lat_flat = latent.float().reshape(latent.shape[0], -1)          # [C, T*H*W]
        lat_mean = lat_flat.mean(dim=1)                                 # [C]
        lat_std  = lat_flat.std(dim=1).clamp(min=1e-6)                  # [C]
        lat_min  = lat_flat.min(dim=1).values                           # [C]
        lat_max  = lat_flat.max(dim=1).values                           # [C]
        z_vec = torch.cat([lat_mean, lat_std, lat_min, lat_max])        # [C*4]

        # Prompt: [1, L, D] → [D * 2] (mean + std over token dim)
        p = prompt_embed.float().squeeze(0)                             # [L, D]
        p_mean = p.mean(dim=0)                                          # [D]
        p_std  = p.std(dim=0).clamp(min=1e-6)                           # [D]
        p_vec = torch.cat([p_mean, p_std])                              # [D*2]

        # ---- 2. Random projection into shared space ---- #
        rng = torch.Generator(device='cpu')
        rng.manual_seed(seed)

        z_dim = z_vec.shape[0]
        p_dim = p_vec.shape[0]

        # Gaussian random matrices, scaled by 1/√d for unit-variance output
        W_z = torch.randn(z_dim, projection_dim, generator=rng, device='cpu',
                          dtype=dtype).to(device) / math.sqrt(projection_dim)
        W_p = torch.randn(p_dim, projection_dim, generator=rng, device='cpu',
                          dtype=dtype).to(device) / math.sqrt(projection_dim)

        z_proj = z_vec @ W_z          # [projection_dim]
        p_proj = p_vec @ W_p          # [projection_dim]

        # ---- 3. Cosine similarity in projected space ---- #
        sim = F.cosine_similarity(z_proj.unsqueeze(0), p_proj.unsqueeze(0))

        return float(sim.item())

    # ------------------------------------------------------------------ #
    #                   Aggregate Section Report                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_section_metrics(
        current_latent: torch.Tensor,
        previous_latent: torch.Tensor | None,
        prompt_embed: torch.Tensor,
        previous_prompt_embed: torch.Tensor | None = None,
    ) -> dict:
        """
        One-call convenience that returns all relevant metrics for a section boundary.

        Parameters
        ----------
        current_latent    : [C, T, H, W]  the newly generated section
        previous_latent   : [C, T, H, W]  the preceding section (None for section 0)
        prompt_embed      : [1, L, D]      current section prompt embedding
        previous_prompt_embed : [1, L, D]  previous section embedding (optional)

        Returns
        -------
        dict  with keys: boundary_ssim, identity_drift, semantic_alignment,
              temporal_coherence
        """
        results = {}

        # Boundary SSIM between last frame of previous and first frame of current
        if previous_latent is not None:
            last_prev  = previous_latent[:, -1, :, :]   # [C, H, W]
            first_curr = current_latent[:, 0, :, :]
            results["boundary_ssim"] = VideoMetrics.calculate_ssim_boundary(last_prev, first_curr)
        else:
            results["boundary_ssim"] = 1.0  # No boundary for first section

        # Identity drift between section embeddings
        if previous_prompt_embed is not None:
            results["identity_drift"] = VideoMetrics.calculate_embedding_drift(
                prompt_embed.mean(dim=1).flatten(),
                previous_prompt_embed.mean(dim=1).flatten(),
            )
        else:
            results["identity_drift"] = 0.0

        # Semantic alignment of generated latent to its prompt
        results["semantic_alignment"] = VideoMetrics.calculate_semantic_alignment(
            current_latent, prompt_embed,
        )

        # Temporal coherence within this section
        results["temporal_coherence"] = VideoMetrics.calculate_temporal_coherence(
            current_latent,
        )

        return results
