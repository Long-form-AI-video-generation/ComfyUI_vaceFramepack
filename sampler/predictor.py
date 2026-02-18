"""
Low-level prediction logic for FramePack video generation.

Extracted from `nodes/samplers.py` — contains the classifier-free guidance
prediction step that runs the model twice (conditional + unconditional)
and blends the outputs.
"""

import torch
import math
import inspect

from comfy import model_management as mm


def predict_with_cfg(
    *,
    latent: torch.Tensor,
    cfg_scale: float,
    text_embeds: dict,
    timestep: torch.Tensor,
    idx: int,
    model_wrapper,
    vace_data,
    seq_len: int,
    freqs: torch.Tensor,
    device: torch.device,
    cache_state: list,
    framepack_history: list | None = None,
) -> tuple[torch.Tensor, list]:
    """
    Perform a single classifier-free guidance prediction step.

    Parameters
    ----------
    latent : Tensor [C, T, H, W]
        Current noisy latent.
    cfg_scale : float
        Classifier-free guidance strength.
    text_embeds : dict
        Must contain ``"prompt_embeds"`` and ``"negative_prompt_embeds"``.
    timestep : Tensor
        Current diffusion timestep.
    idx : int
        Step index (0-based).
    model_wrapper : nn.Module
        The diffusion model.
    vace_data : list | None
        Optional VACE conditioning data.
    seq_len : int
        Sequence length for the transformer.
    freqs : Tensor
        Precomputed RoPE frequencies.
    device : torch.device
    cache_state : list[Any, Any]
        Mutable two-element list ``[cond_cache, uncond_cache]``.
        Updated in-place **and** returned.
    framepack_history : list | None
        Historical latent chunks for MoC routing.

    Returns
    -------
    (noise_pred, cache_state)
        ``noise_pred`` is [C, T, H, W], ``cache_state`` is the updated pair.
    """
    dtype = torch.float32
    latent = latent.to(dtype)

    with torch.autocast(device_type=mm.get_autocast_device(device), dtype=dtype):
        # Base model parameters
        base_params = {
            "seq_len": seq_len,
            "device": device,
            "freqs": freqs,
            "t": timestep.to(device),
            "current_step": idx,
            "nag_params": text_embeds.get("nag_params", {}),
            "nag_context": text_embeds.get("nag_prompt_embeds", None),
            "ref_target_masks": None,
        }

        # Conditionally pass framepack_history if the model supports it
        try:
            forward_method = getattr(model_wrapper, "forward", None)
            if forward_method:
                forward_params = inspect.signature(forward_method).parameters
                if "framepack_history" in forward_params:
                    base_params["framepack_history"] = framepack_history
                elif framepack_history is not None and idx == 0:
                    print(
                        "WARNING: WanModel does not accept 'framepack_history'. "
                        "Please restart ComfyUI to reload the updated model definition."
                    )
        except Exception as e:
            print(f"WARNING: Could not inspect model signature: {e}")

        current_step_percentage = idx / 30

        # ── Conditional pass ──
        noise_pred_cond, cache_state_cond = model_wrapper(
            [latent],
            context=text_embeds["prompt_embeds"],
            y=None,
            clip_fea=None,
            is_uncond=False,
            current_step_percentage=current_step_percentage,
            pred_id=cache_state[0],
            vace_data=vace_data,
            attn_cond=None,
            **base_params,
        )
        noise_pred_cond = noise_pred_cond[0]

        # Skip unconditional when cfg == 1.0
        if math.isclose(cfg_scale, 1.0):
            cache_state[0] = cache_state_cond
            cache_state[1] = None
            return noise_pred_cond, cache_state

        # ── Unconditional pass ──
        noise_pred_uncond, cache_state_uncond = model_wrapper(
            [latent],
            context=text_embeds["negative_prompt_embeds"],
            y=None,
            clip_fea=None,
            is_uncond=True,
            current_step_percentage=current_step_percentage,
            pred_id=cache_state[1],
            vace_data=vace_data,
            attn_cond=None,
            **base_params,
        )
        noise_pred_uncond = noise_pred_uncond[0]

        # ── Apply CFG ──
        noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

        cache_state[0] = cache_state_cond
        cache_state[1] = cache_state_uncond

        return noise_pred, cache_state
