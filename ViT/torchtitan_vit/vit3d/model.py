"""3D Vision Transformer encoder for the ALCF-4 AI benchmark.

The model reuses TorchTitan's ``Decoder`` so it inherits every parallelism the
decoder supports (FSDP/HSDP, DDP, TP + sequence parallel, PP, CP) without
changing TorchTitan. Only the two ends and the attention kernel differ from a
language model:

- ``tok_embeddings`` is a patch-embedding ``Linear`` over flattened
  ``C * p^3`` voxel patches instead of a token lookup table.
- ``lm_head`` is a per-patch regression head back to ``C * p^3`` voxels.
- Attention is bidirectional (non-causal) SDPA; see ``attention.py``.

Training objective: denoising reconstruction. The dataloader delivers clean
patches; ``preprocess_inputs`` adds Gaussian noise on the device and the model
regresses the clean patches with an MSE loss. Every token is a target, so the
loss shards with CP and needs no pooling across ranks.

Shape suffixes: T = packed tokens, F = voxels per patch (C * p^3), D = model dim.
"""

from dataclasses import dataclass, field
from typing import Any

import spmd_types as spmd
import torch
import torch_remat as remat
from spmd_types import SpmdType
from torch import nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed import context_parallel
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
    residual_activation_placement,
    token_id_placement,
)
from torchtitan.models.common.flops import quadratic_attention_flops_per_token
from torchtitan.models.common.linear import Linear

__all__ = [
    "PatchEmbedding",
    "ViT3DModel",
    "ViT3DTransformerBlock",
    "head_norm_sharding_config",
    "patch_embedding_sharding_config",
    "regression_head_sharding_config",
    "vit_input_sharding",
]


def _patch_placement() -> SpmdType:
    """``(T, F)`` patches: DP/CP shard tokens, identical on every TP rank."""
    return dense_activation_placement(tp=spmd.I, cp=spmd.S(0))


def vit_input_sharding() -> dict[str, SpmdType]:
    """Layouts of the tensors ``preprocess_inputs`` hands to the model and loss."""
    return {
        "input": _patch_placement(),
        "labels": _patch_placement(),
        "positions": token_id_placement(),
    }


def patch_embedding_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Patch embedding computed redundantly on each TP rank.

    The output enters the residual layout; under sequence parallelism that is a
    local slice of the token dimension, whose backward all-gathers the gradient
    back to the full, TP-invariant gradient, so the weight stays ``I@TP``.
    """
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        },
        in_src_shardings={"input": _patch_placement()},
        out_src_shardings=_patch_placement(),
        out_dst_shardings=residual_activation_placement(enable_sp=enable_sp),
    )


def head_norm_sharding_config(*, enable_sp: bool) -> ShardingConfig:
    """Final norm that gathers the residual stream to ``I@TP`` for the head.

    Gathering to I (not R) makes the backward a local slice: every TP rank
    computes the same loss, so its gradient is already the full gradient and
    must not be summed across TP.
    """
    residual = residual_activation_placement(enable_sp=enable_sp)
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.R if enable_sp else spmd.I)
        },
        in_src_shardings={"input": residual},
        out_src_shardings=residual,
        out_dst_shardings=_patch_placement(),
    )


def regression_head_sharding_config() -> ShardingConfig:
    """Per-patch regression head, computed redundantly on each TP rank."""
    return ShardingConfig(
        state_shardings={
            "weight": dense_param_placement(tp=spmd.I),
            "bias": dense_param_placement(tp=spmd.I),
        },
        in_src_shardings={"input": _patch_placement()},
        out_src_shardings=_patch_placement(),
    )


class PatchEmbedding(Linear):
    """Linear patch embedding that casts voxel patches to the compute dtype.

    Token-id inputs of a language model need no cast; float patches arrive in
    the loader dtype while FSDP mixed precision computes in bf16.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input.to(self.weight.dtype))


class ViT3DTransformerBlock(TransformerBlock):
    """Pre-norm block: x + attn(norm(x)), then h + ffn(norm(h))."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        pass

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_metadata: None,
        positions: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        attn_out = self.attention(self.attention_norm(x), attention_metadata, positions)
        # The residual add reads the attention output with bare ops.
        remat.recompute_needs_tensor(attn_out)
        h = x + attn_out
        ffn_out = self.feed_forward(self.ffn_norm(h))
        # Trailing add saves nothing for backward, so it is never recomputed.
        return remat.region(
            torch.add, self.remat_region_name("ffn_residual"), recompute=False
        )(h, ffn_out)


class ViT3DModel(Decoder):
    """Encoder-only 3D ViT: patch embedding -> bidirectional blocks -> head."""

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        in_channels: int
        volume_shape: tuple[int, int, int]
        """Input volume ``(D, H, W)`` in voxels."""
        patch_size: int
        """Edge length p of the cubic patches; tokens = D * H * W / p^3."""
        noise_std: float = 0.1
        """Std of the Gaussian noise added to inputs for the denoising target."""
        local_compile: LocalCompileConfig = field(
            default_factory=lambda: LocalCompileConfig(regions=["loss", "rope"])
        )

        def __post_init__(self) -> None:
            Decoder.Config.__post_init__(self)
            if any(edge % self.patch_size != 0 for edge in self.volume_shape):
                raise ValueError(
                    f"volume_shape {self.volume_shape} must be divisible by "
                    f"patch_size {self.patch_size}."
                )
            if self.max_context_length != self.num_tokens_per_sample:
                raise ValueError(
                    f"max_context_length ({self.max_context_length}) must equal "
                    f"tokens per sample ({self.num_tokens_per_sample})."
                )

        @property
        def num_tokens_per_sample(self) -> int:
            d, h, w = self.volume_shape
            return (d // self.patch_size) * (h // self.patch_size) * (
                w // self.patch_size
            )

        @property
        def patch_dim(self) -> int:
            return self.in_channels * self.patch_size**3

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            """Model FLOPs per token for training (forward + backward).

            ``6 * nparams`` covers every per-token matmul, including the patch
            embedding and head. Attention adds ``6 * H * (K + V) * S`` per layer;
            it is non-causal, so the full ``S`` is attended with no discount.
            Recomputation (activation checkpointing, flash backward) is not
            counted, so the resulting TFLOP/s is model FLOP/s.
            """
            nparams = sum(p.numel() for p in model.parameters())
            attention_flops = 0
            for layer in self.layers:
                attention = layer.attention
                head_dim = attention.dim // attention.n_heads
                attention_flops += quadratic_attention_flops_per_token(
                    num_heads=attention.n_heads,
                    qk_head_dim=head_dim,
                    v_head_dim=head_dim,
                    seq_len=seq_len,
                )
            return nparams, 6 * nparams + attention_flops

    def preprocess_inputs(
        self,
        input_dict: dict[str, Any],
        *,
        parallelism_context: ParallelismContext,
        parallelism: ParallelismConfig,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Noise the clean patches, add positions, CP-shard and annotate."""
        del kwargs
        if input_dict.pop("loss_kwargs", None):
            raise ValueError("ViT3DModel does not take per-microbatch loss kwargs.")
        if parallelism.context_parallel_load_balancer is not None:
            raise ValueError(
                "Bidirectional attention has no CP load imbalance; set "
                "context_parallel_load_balancer to None."
            )
        clean_TF = input_dict.pop("labels")
        draw_ids: list[int] = input_dict.pop("draw_ids")
        num_tokens = clean_TF.shape[0]
        seq_len = self.config.num_tokens_per_sample
        with torch.no_grad():
            # Seed each sample's noise by its draw id so every DP/CP/TP/PP
            # layout corrupts a sample identically (bitwise-comparable losses).
            noise_TF = torch.cat(
                [
                    torch.randn(
                        (seq_len, clean_TF.shape[1]),
                        generator=torch.Generator(clean_TF.device).manual_seed(
                            draw_id
                        ),
                        device=clean_TF.device,
                        dtype=clean_TF.dtype,
                    )
                    for draw_id in draw_ids
                ]
            )
            noisy_TF = clean_TF + self.config.noise_std * noise_TF
            positions_T = torch.arange(
                seq_len, device=clean_TF.device, dtype=torch.int64
            ).repeat(num_tokens // seq_len)

        batch = {"input": noisy_TF, "labels": clean_TF, "positions": positions_T}
        input_shardings = vit_input_sharding()
        if parallelism_context.cp_enabled:
            batch = context_parallel.shard_tensors(
                batch, input_shardings=input_shardings, permutation=None
            )
        batch = annotate_input_spmd_types(parallelism_context, batch, input_shardings)
        return batch["input"], batch["labels"], {"positions": batch["positions"]}, {}
