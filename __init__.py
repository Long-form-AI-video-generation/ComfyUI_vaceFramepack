from .nodes.samplers import WanVACEVideoFramepackSampler2
from .nodes.wan_context_node import WanVideoContextSelector

# Node registration
NODE_CLASS_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": WanVACEVideoFramepackSampler2,
    "WanVideoContextSelector": WanVideoContextSelector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVACEVideoFramepackSampler2": "WanVACE FramePack Sampler 2",
    "WanVideoContextSelector": "WanVideo Context Selector",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']