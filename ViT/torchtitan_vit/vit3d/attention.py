"""Bidirectional (non-causal) attention kernels for the 3D ViT benchmark.

Every sample is one fixed-length sequence of ``num_tokens_per_sample`` patch
tokens, so no document masks or varlen metadata are needed. The kernels call
``F.scaled_dot_product_attention`` with ``is_causal=False``, which dispatches to
each vendor's fused kernel (cuDNN/FlashAttention on CUDA, SYCL-TLA flash on XPU,
CK/AOTriton on ROCm).

Shape suffixes: B = samples, S = tokens per sample, T = packed tokens (B * S),
H = heads, K = qk head dim, V = v head dim.
"""

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.context_parallel import cp_all_gather_tensors
from torchtitan.models.common.attention import InnerAttention
from torchtitan.models.common.attention.cp_attention import (
    CPInnerAttention,
    UlyssesInnerAttention,
)

__all__ = [
    "AllGatherKVBidirectionalInnerAttention",
    "BidirectionalInnerAttention",
    "UlyssesBidirectionalInnerAttention",
]


class BidirectionalInnerAttention(InnerAttention):
    """Full non-causal SDPA over fixed-length samples packed along tokens."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        num_tokens_per_sample: int
        """Sequence length S of one sample; the packed token count is B * S."""

        @property
        def is_full_attention(self) -> bool:
            return True

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.num_tokens_per_sample = config.num_tokens_per_sample

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_metadata: None = None,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del attention_metadata, kwargs
        num_kv_tokens = k_THK.shape[0]
        if num_kv_tokens % self.num_tokens_per_sample != 0:
            raise ValueError(
                f"Key tokens ({num_kv_tokens}) must be a multiple of "
                f"num_tokens_per_sample ({self.num_tokens_per_sample})."
            )
        num_samples = num_kv_tokens // self.num_tokens_per_sample
        # Queries may be a CP shard of one sample (all-gather K/V CP), so their
        # per-sample length is derived instead of assumed to be S.
        q_BHSK = q_THK.unflatten(0, (num_samples, -1)).transpose(1, 2)
        k_BHSK = k_THK.unflatten(0, (num_samples, -1)).transpose(1, 2)
        v_BHSV = v_THV.unflatten(0, (num_samples, -1)).transpose(1, 2)
        out_BHSV = F.scaled_dot_product_attention(
            q_BHSK, k_BHSK, v_BHSV, is_causal=False, scale=scale
        )
        return out_BHSV.transpose(1, 2).flatten(0, 1)


class UlyssesBidirectionalInnerAttention(
    UlyssesInnerAttention[None], BidirectionalInnerAttention
):
    """Ulysses CP: all-to-all tokens -> heads, full attention, heads -> tokens.

    Each layer issues 3 all-to-alls (q, k, v) before attention and 1 after in
    forward, and the same in backward. ``n_heads`` must be divisible by
    ``tensor_parallel_degree * context_parallel_degree``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(UlyssesInnerAttention.Config, BidirectionalInnerAttention.Config):
        pass


class AllGatherKVBidirectionalInnerAttention(
    CPInnerAttention[None, None], BidirectionalInnerAttention
):
    """All-gather K/V CP: local query shard attends to the gathered sequence.

    Forward all-gathers K and V over the CP axis; backward reduce-scatters their
    gradients. Non-causal attention has no load imbalance, so contiguous token
    shards need no permutation. Each CP group must hold exactly one sample per
    microbatch so a contiguous query shard never spans two samples.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CPInnerAttention.Config, BidirectionalInnerAttention.Config):
        reduce_dtype: Literal["float32", "bfloat16"] = "float32"
        """Dtype of the backward K/V gradient reduce-scatter."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.reduce_dtype = TORCH_DTYPE_MAP[config.reduce_dtype]

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: None,
        *,
        permutation: torch.Tensor | None,
    ) -> None:
        if permutation is not None:
            raise ValueError(
                "Bidirectional all-gather K/V CP uses contiguous token shards; "
                "set context_parallel_load_balancer to None."
            )
        return attention_metadata

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_THK, v_THV = cp_all_gather_tensors(
            (k_THK, v_THV), reduce_dtype=self.reduce_dtype
        )
        if k_THK.shape[0] != self.num_tokens_per_sample:
            raise ValueError(
                "All-gather K/V CP requires one sample per microbatch per CP "
                f"group, but gathered {k_THK.shape[0]} tokens for samples of "
                f"{self.num_tokens_per_sample} tokens."
            )
        return super().forward(q_THK, k_THK, v_THV, **kwargs)
