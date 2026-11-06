"""3D Vision Transformer benchmark on TorchTitan."""

from .flavors import build_model_config, MODEL_FLAVORS
from .model import ViT3DModel

__all__ = ["MODEL_FLAVORS", "ViT3DModel", "build_model_config"]
