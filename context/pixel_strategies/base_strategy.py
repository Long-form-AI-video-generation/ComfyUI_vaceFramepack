from abc import ABC, abstractmethod
import torch

class ContextStrategy(ABC):
    @abstractmethod
    def select(self, history_images: torch.Tensor, context_size: int, **kwargs) -> torch.Tensor:
        ...
