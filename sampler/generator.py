"""
Core FramePack generation loop.

Extracted from `nodes/samplers.py` — contains the multi-section generation
algorithm that orchestrates encoding → denoising → accumulation for each
video section.
"""

import torch
import math
import gc

from comfy import model_management as mm

from ..utils.vae import (
    SchedulerFactory,
    RoPEEmbeddings,
    ReferenceImageProcessor,
    VAE_STRIDE,
)
from ..context.builder import ContextBuilder, MaskGenerator
from ..context.strategies import FramePackCompressor

from .predictor import predict_with_cfg


# ── Constants ──────────────────────────────────────────────────────────── #

LATENT_WINDOW = 41
GENERATION_FRAMES = 30
CONTEXT_FRAMES = 11
INITIAL_FRAMES = 81


# ── Public API ─────────────────────────────────────────────────────────── #

def generate_with_framepack_multi(
    *,
    # model / pipeline state
    model_wrapper,
    vae_processor,
    benchmark_manager,
    # per-section data
    section_text_embeds,
    section_prompts,
    # input conditioning
    input_frames,
    input_masks,
    ref_images,
    # generation parameters
    width,
    height,
    num_frames,
    shift,
    scheduler_name,
    mode,
    steps,
    cfg,
    seed,
    sigmas,
    # device control
    device,
    offload_device,
    force_offload,
):
    """
    Core FramePack generation algorithm with multi-prompt support.

    Returns
    -------
    tuple[torch.Tensor, list[list | None]]
        (final_latent_cpu, cache_state)
        ``final_latent_cpu`` is [C, T_total, H, W] on CPU.
    """
    from diffusers.schedulers import DEISMultistepScheduler
    from ..wanvideo.utils.basic_flowmatch import FlowMatchScheduler
    from comfy.utils import ProgressBar

    vae_dtype = torch.float32
    all_generated_latents = []
    accumulated_latents = []
    total_output_frames = 0

    # Mutable cache state shared across predict calls
    cache_state = [None, None]

    # FramePackCompressor (only for "frame" mode)
    frame_compressor = None
    if mode == "frame":
        frame_compressor = FramePackCompressor()
        print("Initialized FramePackCompressor for frame-level compression")

    num_sections = (
        1 if num_frames <= INITIAL_FRAMES
        else math.ceil(num_frames / INITIAL_FRAMES)
    )

    for section in range(num_sections):
        print(f"\n[Section {section + 1}/{num_sections}]")
        print(f"Using prompt: {section_prompts[section][:100]}...")

        text_embeds = section_text_embeds[section]

        # ── PHASE 1: ENCODING ──────────────────────────────────────── #
        benchmark_manager.benchmark_section(section, "encoding")

        framepack_history = []
        vace_data = None

        if section == 0:
            # Initial section setup
            input_frames = torch.zeros(
                1, 3, INITIAL_FRAMES, height, width,
                device=device, dtype=vae_dtype,
            )
            input_masks = torch.ones_like(input_frames, device=device, dtype=vae_dtype)
            input_frames = [(f * 2 - 1) for f in input_frames]

            # Process reference images if provided
            if ref_images is not None:
                ref_images = ReferenceImageProcessor.process_reference_images(
                    ref_images, width, height, device, vae_dtype,
                )

            target_shape = (
                16,
                (INITIAL_FRAMES - 1) // VAE_STRIDE[0] + 1,
                height // VAE_STRIDE[1],
                width // VAE_STRIDE[2],
            )
        else:
            # Build context based on mode
            if mode == "sparse":
                context_latent = ContextBuilder.build_hierarchical_context(
                    accumulated_latents, section,
                )
                hierarchical_frames = ContextBuilder.pick_context(context_latent, section)

                input_frames = vae_processor.decode_latent([hierarchical_frames], None)
                input_frames[0] = input_frames[0].expand(3, -1, -1, -1)

                input_masks = MaskGenerator.create_temporal_blend_mask(
                    input_frames[0].shape, section, device,
                )

                z0 = vae_processor.encode_frames(
                    input_frames, ref_images=None, masks=input_masks, tiled_vae=False,
                )
                m0 = vae_processor.encode_masks(input_masks, ref_images=None)
                z = vae_processor.combine_latent(z0, m0)

                vace_data = [{
                    "context": z,
                    "scale": [1.0] * (steps + 1),
                    "start": 0.0,
                    "end": 1.0,
                    "seq_len": math.ceil(
                        (width // VAE_STRIDE[2] * height // VAE_STRIDE[1]) / 4 * LATENT_WINDOW,
                    ),
                }]

            elif mode == "frame":
                compressed_history = frame_compressor.compress_history(accumulated_latents)

                context_latent = frame_compressor.select_context_frames(
                    compressed_history,
                    num_context_frames=CONTEXT_FRAMES,
                    add_generation_frames=True,
                ).to(device)

                # Channel adaptation
                if (
                    hasattr(model_wrapper, "vace_in_dim")
                    and model_wrapper.vace_in_dim != context_latent.shape[0]
                ):
                    target_dim = model_wrapper.vace_in_dim
                    current_dim = context_latent.shape[0]
                    if target_dim > current_dim and target_dim % current_dim == 0:
                        repeat_factor = target_dim // current_dim
                        print(
                            f"Adapting context latent channels: "
                            f"{current_dim} → {target_dim} (repeat {repeat_factor}×)"
                        )
                        context_latent = context_latent.repeat(repeat_factor, 1, 1, 1)
                    else:
                        print(
                            f"WARNING: Channel mismatch in Frame mode: "
                            f"got {current_dim}, model expects {target_dim}"
                        )

                z = [context_latent]
                vace_data = [{
                    "context": z,
                    "scale": [1.0] * (steps + 1),
                    "start": 0.0,
                    "end": 1.0,
                    "seq_len": math.ceil(
                        (width // VAE_STRIDE[2] * height // VAE_STRIDE[1]) / 4 * LATENT_WINDOW,
                    ),
                }]

            else:  # mode == "moc" (default)
                for lat in accumulated_latents:
                    framepack_history.append(lat.unsqueeze(0))

            target_shape = (
                16,
                (INITIAL_FRAMES - 1) // VAE_STRIDE[0] + 1,
                height // VAE_STRIDE[1],
                width // VAE_STRIDE[2],
            )
            ref_images = None

        benchmark_manager.benchmark_section(section, "encoding")  # End

        # ── PHASE 2: DENOISING ─────────────────────────────────────── #
        benchmark_manager.benchmark_section(section, "denoising")

        sample_scheduler = SchedulerFactory.create_scheduler(
            scheduler_name, steps, shift, device, sigmas,
        )
        timesteps = sample_scheduler.timesteps

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            seed if seed != -1 else torch.randint(0, 2**32, (1,)).item()
        )

        has_ref = ref_images is not None
        noise = torch.randn(
            target_shape[0],
            target_shape[1] + (1 if has_ref else 0),
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device="cpu",
            generator=generator,
        )

        latent = noise.to(device)

        seq_len = math.ceil(
            (noise.shape[2] * noise.shape[3]) / 4 * noise.shape[1]
        )
        freqs = RoPEEmbeddings.setup_rope_embeddings(model_wrapper, latent.shape[1])
        num_steps = len(timesteps)

        cfg_list = cfg if isinstance(cfg, list) else [cfg] * (steps + 1)

        pbar = ProgressBar(steps)

        mm.soft_empty_cache()
        gc.collect()

        cache_state = [None, None]

        for idx, t in enumerate(timesteps):
            print(idx + 1, "of", num_steps)
            timestep = torch.tensor([t]).to(device)

            noise_pred, cache_state = predict_with_cfg(
                latent=latent,
                cfg_scale=cfg_list[idx],
                text_embeds=text_embeds,
                timestep=timestep,
                idx=idx,
                model_wrapper=model_wrapper,
                vace_data=vace_data,
                seq_len=seq_len,
                freqs=freqs,
                device=device,
                cache_state=cache_state,
                framepack_history=framepack_history,
            )

            step_args = {"generator": generator}
            if isinstance(sample_scheduler, (DEISMultistepScheduler, FlowMatchScheduler)):
                step_args.pop("generator", None)

            latent = sample_scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                latent.unsqueeze(0),
                **step_args,
            )[0].squeeze(0)

            pbar.update(1)

            if force_offload and idx % 10 == 0:
                mm.soft_empty_cache()

        benchmark_manager.benchmark_section(section, "denoising")  # End

        # ── PHASE 3: ACCUMULATION ──────────────────────────────────── #
        benchmark_manager.benchmark_section(section, "accumulation")

        if section == 0:
            latent_out = latent[:, 1:, :, :] if ref_images is not None else latent
            accumulated_latents.append(latent_out)
            all_generated_latents.append(latent_out)
        else:
            new = latent[:, -GENERATION_FRAMES:, :, :]
            accumulated_latents.append(new)

            new_content = new[:, CONTEXT_FRAMES:, :, :]
            all_generated_latents.append(new_content)

            frames_added = new_content.shape[1]
            total_output_frames += frames_added
            print(f"Added {frames_added} frames (total: {total_output_frames})")

        benchmark_manager.benchmark_section(section, "accumulation")  # End

        # Clean up section temporaries
        if "noise_pred" in dir():
            del latent, noise_pred
        mm.soft_empty_cache()
        gc.collect()

    # Final offload
    if force_offload:
        model_wrapper.to(offload_device)
        mm.soft_empty_cache()
        gc.collect()

    final_latent = torch.cat(all_generated_latents, dim=1)
    return final_latent.cpu(), cache_state
