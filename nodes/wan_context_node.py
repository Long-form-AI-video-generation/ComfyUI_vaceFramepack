import torch
from ..context.selector_factory import SelectorFactory

class WanVideoContextSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "selection_mode": (["contiguous", "moc"], {"default": "contiguous"}),
                "context_size": ("INT", {"default": 16, "min": 1, "max": 128}),
            },
            "optional": {
                "clip_vision": ("CLIP_VISION",),
                "text_encoder": ("WANTEXTENCODER",),
                "current_prompt": ("STRING", {"multiline": True}),
                "contiguous_size": ("INT", {"default": 4, "min": 0, "max": 32, "tooltip": "Recent frames to always preserve"}),
                "text_weight": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "0=Visual Only, 1=prompt Only"}),
                "diversity_radius": ("INT", {"default": 16, "min": 0, "max": 128, "tooltip": "Suppression window to prevent clumping (Soft-NMS)"}),
                "similarity_threshold": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05, "tooltip": "Minimum score to include a frame"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("context_images",)
    FUNCTION = "select_context"
    CATEGORY = "WanVideo/Context"

    def select_context(self, images, selection_mode, context_size, 
                      clip_vision=None, text_encoder=None, current_prompt=None,
                      contiguous_size=4, text_weight=0.0, diversity_radius=0, similarity_threshold=0.0):
        
        if images is None:
            return (None,)

        strategy = SelectorFactory.get_strategy(selection_mode)
        
        selected_frames = strategy.select(
            history_images=images, 
            context_size=context_size, 
            clip_vision=clip_vision,
            text_encoder=text_encoder,
            current_prompt=current_prompt,
            contiguous_size=contiguous_size,
            text_weight=text_weight,
            diversity_radius=diversity_radius,
            similarity_threshold=similarity_threshold
        )
        
        return (selected_frames,)
