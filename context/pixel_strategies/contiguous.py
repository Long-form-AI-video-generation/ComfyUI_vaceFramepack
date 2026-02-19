import torch
from .base_strategy import ContextStrategy
from ...utils.tensor_utils import safe_slice

class ContiguousStrategy(ContextStrategy):
    def select(self, history_images: torch.Tensor, context_size: int, **kwargs) -> torch.Tensor:
        total = history_images.shape[0]
        start = max(0, total - context_size)
        return safe_slice(history_images, start, total)
