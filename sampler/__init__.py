"""
Sampler sub-package for FramePack video generation.

Contains the core generation loop and prediction logic as standalone functions:
- ``generate_with_framepack_multi`` — multi-section generation orchestrator
- ``predict_with_cfg`` — single-step classifier-free guidance prediction
"""

from .generator import generate_with_framepack_multi
from .predictor import predict_with_cfg

__all__ = [
    "generate_with_framepack_multi",
    "predict_with_cfg",
]
