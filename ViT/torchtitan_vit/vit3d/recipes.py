"""TorchTitan recipes for the 3D ViT benchmark.

Run any recipe with ``--module vit3d.recipes --config <function>``.

``vit3d_scaling`` is the benchmark entry point: problem size and every
parallelism degree come from ``VIT_*`` environment variables so a job script
can sweep them without editing code (see README.md). The ``vit3d_debug_*``
recipes are small fixed configurations used for functional tests.
"""

import os
from dataclasses import replace
from typing import Literal

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import MSELoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.configs import DebugConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import (
    DefaultActivationRematConfig,
    FullActivationRematConfig,
)
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from .data import NoTokenizer, VolumeDataLoader
from .flavors import build_model_config, CPBackend

__all__ = [
    "vit3d",
    "vit3d_debug",
    "vit3d_debug_4d",
    "vit3d_debug_cp_allgather_kv",
    "vit3d_debug_cp_ulysses",
    "vit3d_debug_ddp",
    "vit3d_debug_fsdp",
    "vit3d_debug_hsdp",
    "vit3d_debug_one_sample",
    "vit3d_debug_pp",
    "vit3d_debug_seed_checkpoint",
    "vit3d_debug_tp",
    "vit3d_scaling",
]


def vit3d(
    *,
    flavor: str,
    volume_shape: tuple[int, int, int],
    patch_size: int,
    parallelism: ParallelismConfig,
    samples_per_microbatch: int = 1,
    in_channels: int = 1,
    cp_backend: CPBackend = "ulysses",
    steps: int = 50,
    lr: float = 3e-4,
    activation_remat: Literal["none", "default", "full"] = "default",
    dataloader: VolumeDataLoader.Config | None = None,
) -> Trainer.Config:
    """Assemble a full trainer config for one problem size and parallelism."""
    model = build_model_config(
        flavor,
        volume_shape=volume_shape,
        patch_size=patch_size,
        in_channels=in_channels,
        parallelism=parallelism,
        cp_backend=cp_backend,
    )
    seq_len = model.num_tokens_per_sample
    if dataloader is None:
        dataloader = VolumeDataLoader.Config(
            volume_shape=volume_shape,
            patch_size=patch_size,
            in_channels=in_channels,
        )
    remat = {
        "none": None,
        "default": DefaultActivationRematConfig(),
        "full": FullActivationRematConfig(),
    }[activation_remat]
    return Trainer.Config(
        model=model,
        tokenizer=NoTokenizer.Config(),
        loss=MSELoss.Config(),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=lr)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=max(1, steps // 10),
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.1,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=samples_per_microbatch * seq_len,
            max_context_length=seq_len,
            steps=steps,
            # Input shapes are fixed, but graph capture would also capture the
            # timed host-to-device copy; keep the step eager and portable.
            disable_accelerator_graphs=True,
        ),
        dataloader=dataloader,
        parallelism=parallelism,
        metrics=MetricsProcessor.Config(log_freq=1),
        activation_remat=remat,
        checkpointer=None,
        validator=None,
    )


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def vit3d_scaling() -> Trainer.Config:
    """Benchmark entry point; every knob is a ``VIT_*`` environment variable.

    Problem: VIT_FLAVOR (large), VIT_VOLUME (256, cube edge or ``DxHxW``),
    VIT_PATCH (8), VIT_CHANNELS (1), VIT_SAMPLES_PER_MICROBATCH (1),
    VIT_STEPS (50), VIT_REMAT (default | full | none).

    Parallelism (product must equal the world size): VIT_DP_REPLICATE (1),
    VIT_DP_SHARD (-1 = remaining ranks), VIT_TP (1), VIT_SP (1 = sequence
    parallel with TP), VIT_PP (1), VIT_PP_SCHEDULE (1F1B),
    VIT_PP_MICROBATCHES (VIT_PP), VIT_CP (1), VIT_CP_BACKEND (ulysses |
    allgather_kv).

    Data: VIT_DATA (hdf5 | synthetic), VIT_DATA_GLOB (required for hdf5),
    VIT_DATA_KEY (volume), VIT_DATA_DTYPE (float32 | bfloat16),
    VIT_NUM_WORKERS (8), VIT_PREFETCH (4), VIT_MEASURE_H2D (1 | 0).

    VIT_SYNTHETIC_POOL (4 volumes per worker, cycled; 0 regenerates each one).
    """
    edges = os.environ.get("VIT_VOLUME", "256").lower().split("x")
    volume_shape = (
        (int(edges[0]),) * 3 if len(edges) == 1 else tuple(int(e) for e in edges)
    )
    patch_size = _env_int("VIT_PATCH", 8)
    in_channels = _env_int("VIT_CHANNELS", 1)
    pp = _env_int("VIT_PP", 1)
    parallelism = ParallelismConfig(
        data_parallel_replicate_degree=_env_int("VIT_DP_REPLICATE", 1),
        data_parallel_shard_degree=_env_int("VIT_DP_SHARD", -1),
        tensor_parallel_degree=_env_int("VIT_TP", 1),
        enable_sequence_parallel=bool(_env_int("VIT_SP", 1)),
        pipeline_parallel_degree=pp,
        pipeline_parallel_schedule=os.environ.get("VIT_PP_SCHEDULE", "1F1B"),
        num_pp_microbatches=_env_int("VIT_PP_MICROBATCHES", pp),
        context_parallel_degree=_env_int("VIT_CP", 1),
    )
    dataloader = VolumeDataLoader.Config(
        volume_shape=volume_shape,  # pyrefly: ignore [bad-argument-type]
        patch_size=patch_size,
        in_channels=in_channels,
        source=os.environ.get("VIT_DATA", "hdf5"),  # pyrefly: ignore
        path_glob=os.environ.get("VIT_DATA_GLOB", ""),
        dataset_key=os.environ.get("VIT_DATA_KEY", "volume"),
        dtype=os.environ.get("VIT_DATA_DTYPE", "float32"),  # pyrefly: ignore
        num_workers=_env_int("VIT_NUM_WORKERS", 8),
        prefetch_factor=_env_int("VIT_PREFETCH", 4),
        synthetic_pool_size=_env_int("VIT_SYNTHETIC_POOL", 4),
        measure_host_to_device=bool(_env_int("VIT_MEASURE_H2D", 1)),
    )
    return vit3d(
        flavor=os.environ.get("VIT_FLAVOR", "large"),
        volume_shape=volume_shape,  # pyrefly: ignore [bad-argument-type]
        patch_size=patch_size,
        in_channels=in_channels,
        parallelism=parallelism,
        samples_per_microbatch=_env_int("VIT_SAMPLES_PER_MICROBATCH", 1),
        cp_backend=os.environ.get("VIT_CP_BACKEND", "ulysses"),  # pyrefly: ignore
        steps=_env_int("VIT_STEPS", 50),
        activation_remat=os.environ.get("VIT_REMAT", "default"),  # pyrefly: ignore
        dataloader=dataloader,
    )


# ---------------------------------------------------------------------------
# Functional-test recipes: 16^3 volumes, 4^3 patches -> 64 tokens of 64 voxels.
# ---------------------------------------------------------------------------

_DEBUG_VOLUME = (16, 16, 16)
_DEBUG_PATCH = 4


def vit3d_debug(
    parallelism: ParallelismConfig | None = None,
    *,
    cp_backend: CPBackend = "ulysses",
    samples_per_microbatch: int = 2,
) -> Trainer.Config:
    """Single-rank baseline that the parallel variants must match."""
    if parallelism is None:
        parallelism = ParallelismConfig()
    config = vit3d(
        flavor="debugmodel",
        volume_shape=_DEBUG_VOLUME,
        patch_size=_DEBUG_PATCH,
        parallelism=parallelism,
        samples_per_microbatch=samples_per_microbatch,
        cp_backend=cp_backend,
        steps=10,
        lr=1e-3,
        dataloader=VolumeDataLoader.Config(
            volume_shape=_DEBUG_VOLUME,
            patch_size=_DEBUG_PATCH,
            source="synthetic",
            num_synthetic_samples=256,
            num_workers=0,
            shuffle=False,
        ),
    )
    # Comparable losses across parallel layouts need a fixed seed,
    # deterministic kernels, fp32 everywhere and identical initial weights:
    # sharded layouts initialize each shard separately, so every run loads the
    # seed checkpoint written by ``vit3d_debug_seed_checkpoint``.
    config.debug = DebugConfig(seed=42, deterministic=True)
    config.training = replace(config.training, mixed_precision_param="float32")
    seed_checkpoint = os.environ.get("VIT_SEED_CHECKPOINT")
    if seed_checkpoint:
        config.checkpointer = CheckpointManager.Config(
            initial_load_path=seed_checkpoint, load_only=True
        )
    return config


def vit3d_debug_seed_checkpoint() -> Trainer.Config:
    """Write the debug model's initial weights to ``<output-dir>/checkpoint``."""
    config = vit3d_debug()
    config.checkpointer = CheckpointManager.Config()
    config.create_seed_checkpoint = True
    return config


# Each variant below sees the same two samples per optimizer step as
# ``vit3d_debug`` (data-parallel variants split them across ranks), so all of
# them must reproduce its loss curve.


def vit3d_debug_one_sample() -> Trainer.Config:
    """Baseline for all-gather K/V CP, which needs one sample per microbatch."""
    return vit3d_debug(samples_per_microbatch=1)


def vit3d_debug_fsdp() -> Trainer.Config:
    """FSDP2: all-gather params, reduce-scatter grads (run on 2 ranks)."""
    return vit3d_debug(
        ParallelismConfig(data_parallel_shard_degree=2), samples_per_microbatch=1
    )


def vit3d_debug_ddp() -> Trainer.Config:
    """DDP: gradient all-reduce only (run on 2 ranks)."""
    return vit3d_debug(
        ParallelismConfig(data_parallel_replicate_degree=2, data_parallel_shard_degree=1),
        samples_per_microbatch=1,
    )


def vit3d_debug_hsdp() -> Trainer.Config:
    """HSDP: shard inside a group, all-reduce across groups (run on 4 ranks).

    Four DP ranks see four samples per step; compare with
    ``vit3d_debug(samples_per_microbatch=4)``.
    """
    return vit3d_debug(
        ParallelismConfig(data_parallel_replicate_degree=2, data_parallel_shard_degree=2),
        samples_per_microbatch=1,
    )


def vit3d_debug_tp() -> Trainer.Config:
    """TP + sequence parallel: activation all-gather/reduce-scatter (2 ranks)."""
    return vit3d_debug(
        ParallelismConfig(tensor_parallel_degree=2, data_parallel_shard_degree=1)
    )


def vit3d_debug_pp() -> Trainer.Config:
    """Pipeline parallel: point-to-point activations (run on 2 ranks)."""
    return vit3d_debug(
        ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule="1F1B",
            num_pp_microbatches=2,
            data_parallel_shard_degree=1,
        ),
        samples_per_microbatch=1,
    )


def vit3d_debug_cp_ulysses() -> Trainer.Config:
    """Ulysses CP: all-to-all tokens <-> heads (run on 2 ranks)."""
    return vit3d_debug(
        ParallelismConfig(context_parallel_degree=2, data_parallel_shard_degree=1),
        cp_backend="ulysses",
    )


def vit3d_debug_cp_allgather_kv() -> Trainer.Config:
    """All-gather K/V CP: K/V all-gather, dK/dV reduce-scatter (2 ranks)."""
    return vit3d_debug(
        ParallelismConfig(context_parallel_degree=2, data_parallel_shard_degree=1),
        cp_backend="allgather_kv",
        samples_per_microbatch=1,
    )


def vit3d_debug_4d() -> Trainer.Config:
    """FSDP x CP(Ulysses) x TP on 8 ranks: 2 x 2 x 2."""
    return vit3d_debug(
        ParallelismConfig(
            data_parallel_shard_degree=2,
            context_parallel_degree=2,
            tensor_parallel_degree=2,
        ),
        cp_backend="ulysses",
        samples_per_microbatch=1,
    )
