"""3D ViT model flavors and the sharded config builder."""

from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import ComplexRoPE, Linear, RMSNorm
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.common.decoder_sharding import norm_sharding_config
from torchtitan.models.common.param_init import depth_scaled_std

from .attention import (
    AllGatherKVBidirectionalInnerAttention,
    BidirectionalInnerAttention,
    UlyssesBidirectionalInnerAttention,
)
from .model import (
    head_norm_sharding_config,
    PatchEmbedding,
    patch_embedding_sharding_config,
    regression_head_sharding_config,
    ViT3DModel,
    ViT3DTransformerBlock,
)

__all__ = ["CPBackend", "MODEL_FLAVORS", "ViTFlavor", "build_model_config"]

CPBackend = Literal["ulysses", "allgather_kv"]
"""Context-parallel attention algorithm, used only when CP degree > 1.

- ``ulysses``: all-to-all tokens <-> heads (DeepSpeed-Ulysses). Needs
  ``n_heads % (tp * cp) == 0``; best inside a scale-up domain.
- ``allgather_kv``: all-gather K/V, reduce-scatter dK/dV (Llama 3 style). No
  head constraint; one sample per microbatch per CP group.
"""


@dataclass(frozen=True, kw_only=True, slots=True)
class ViTFlavor:
    dim: int
    n_layers: int
    n_heads: int
    ffn_hidden_dim: int
    """SwiGLU hidden size. 8/3 * dim gives the FLOPs of a 4 * dim GELU MLP."""


def _swiglu_hidden(dim: int, multiple_of: int = 256) -> int:
    hidden = int(8 * dim / 3)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


MODEL_FLAVORS: dict[str, ViTFlavor] = {
    # Small enough for CPU functional tests.
    "debugmodel": ViTFlavor(dim=64, n_layers=4, n_heads=8, ffn_hidden_dim=192),
    # ViT-B / L / H / g widths, the sizes used in the Megatron-DeepSpeed runs.
    "base": ViTFlavor(dim=768, n_layers=12, n_heads=12, ffn_hidden_dim=2048),
    "large": ViTFlavor(dim=1024, n_layers=24, n_heads=16, ffn_hidden_dim=2816),
    "huge": ViTFlavor(dim=1280, n_layers=32, n_heads=16, ffn_hidden_dim=3584),
    "giant": ViTFlavor(dim=1536, n_layers=40, n_heads=24, ffn_hidden_dim=4096),
    # Wider heads count so Ulysses can span a 12-tile Aurora node (cp*tp | 48).
    "large_h48": ViTFlavor(
        dim=1536, n_layers=24, n_heads=48, ffn_hidden_dim=_swiglu_hidden(1536)
    ),
}

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}


def _depth_init(layer_id: int) -> dict:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _inner_attention_config(
    *, parallelism: ParallelismConfig, cp_backend: CPBackend, seq_len: int
):
    if parallelism.context_parallel_degree == 1:
        return BidirectionalInnerAttention.Config(num_tokens_per_sample=seq_len)
    if cp_backend == "ulysses":
        return UlyssesBidirectionalInnerAttention.Config(num_tokens_per_sample=seq_len)
    if cp_backend == "allgather_kv":
        return AllGatherKVBidirectionalInnerAttention.Config(
            num_tokens_per_sample=seq_len
        )
    raise ValueError(f"Unknown cp_backend {cp_backend!r}.")


def build_model_config(
    flavor: str,
    *,
    volume_shape: tuple[int, int, int],
    patch_size: int,
    parallelism: ParallelismConfig,
    in_channels: int = 1,
    cp_backend: CPBackend = "ulysses",
    rope_theta: float = 10000.0,
) -> ViT3DModel.Config:
    """Build a 3D ViT config sharded for ``parallelism``.

    Positions are the raster index of each patch, encoded with 1D RoPE, so the
    positional encoding adds no parameters as the sequence grows.
    """
    spec = MODEL_FLAVORS[flavor]
    d, h, w = volume_shape
    seq_len = (d // patch_size) * (h // patch_size) * (w // patch_size)
    patch_dim = in_channels * patch_size**3
    enable_sp = parallelism.enable_sequence_parallel
    head_dim = spec.dim // spec.n_heads
    inner_attention = _inner_attention_config(
        parallelism=parallelism, cp_backend=cp_backend, seq_len=seq_len
    )
    rope = ComplexRoPE.Config(
        dim=head_dim, max_context_length=seq_len, theta=rope_theta
    )
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)

    layers = []
    for layer_id in range(spec.n_layers):
        layers.append(
            ViT3DTransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=spec.dim,
                    param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=spec.dim,
                    param_init=_NORM_INIT,
                    sharding_config=norm_sharding,
                ),
                attention=make_gqa_config(
                    dim=spec.dim,
                    n_heads=spec.n_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_depth_init(layer_id),
                    inner_attention=inner_attention,
                    rope=rope,
                    enable_sp=enable_sp,
                ),
                feed_forward=make_ffn_config(
                    dim=spec.dim,
                    hidden_dim=spec.ffn_hidden_dim,
                    w13_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
            )
        )

    return ViT3DModel.Config(
        max_context_length=seq_len,
        dim=spec.dim,
        vocab_size=patch_dim,
        in_channels=in_channels,
        volume_shape=volume_shape,
        patch_size=patch_size,
        tok_embeddings=PatchEmbedding.Config(
            in_features=patch_dim,
            out_features=spec.dim,
            bias=True,
            param_init=_LINEAR_INIT,
            sharding_config=patch_embedding_sharding_config(enable_sp=enable_sp),
        ),
        norm=RMSNorm.Config(
            normalized_shape=spec.dim,
            param_init=_NORM_INIT,
            sharding_config=head_norm_sharding_config(enable_sp=enable_sp),
        ),
        lm_head=Linear.Config(
            in_features=spec.dim,
            out_features=patch_dim,
            bias=True,
            param_init=_LINEAR_INIT,
            sharding_config=regression_head_sharding_config(),
        ),
        layers=layers,
    )
