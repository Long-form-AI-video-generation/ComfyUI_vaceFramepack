from .pixel_strategies.contiguous import ContiguousStrategy
from .pixel_strategies.moc import MoCStrategy

class SelectorFactory:
    @staticmethod
    def get_strategy(mode):
        if mode == "contiguous":
            return ContiguousStrategy()
        elif mode == "moc":
            return MoCStrategy()
        else:
            raise ValueError(f"Unknown selection mode: {mode}")
