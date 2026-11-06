"""Volumetric input pipeline: disk -> host patchify -> pinned memory -> device.

The whole pipeline runs every step, from the disk read to the model's compute
and communication:

1. Read: worker processes read whole volumes from HDF5 on disk (or generate
   them in host memory with the ``synthetic`` source).
2. Preprocess: workers cut each volume into ``p^3`` patches and stack the
   microbatch on the host.
3. Microbatches are pinned and copied host -> device.

Measured and logged per rank (``loader_stats_log_freq``):

- ``loader`` line: stage 1 (read ms and GB/s per sample, per worker) and
  stage 2 (preprocess ms per sample), timed inside the workers.
- ``data_rate`` line: samples/s and GB/s the training loop consumes.
- ``host_to_device`` line (``measure_host_to_device``): copy ms and GB/s.
- TorchTitan's ``data_loading(%)``: the share of step time spent waiting on
  this loader.

Every CP and TP rank of one data-parallel rank loads the same full sample and
CP slices it on the device. Reading only each CP rank's slab is future work;
until then the I/O volume per sample is ``cp * tp`` times the sample size.

Shape suffixes: B = samples, S = tokens per sample, F = voxels per patch.
"""

import glob
import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader, Dataset

from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.tools import utils as device_utils

__all__ = [
    "NoTokenizer",
    "VolumeDataLoader",
    "VolumeTrainingMicrobatch",
    "patchify",
]

logger = logging.getLogger(__name__)


def patchify(volume_CDHW: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Cut a ``(C, D, H, W)`` volume into raster-ordered ``(S, C * p^3)`` patches."""
    c, d, h, w = volume_CDHW.shape
    p = patch_size
    patches = volume_CDHW.reshape(c, d // p, p, h // p, p, w // p, p)
    # -> (D/p, H/p, W/p, C, p, p, p): token order is depth-major raster order.
    patches = patches.permute(1, 3, 5, 0, 2, 4, 6)
    return patches.reshape(-1, c * p**3)


class NoTokenizer(BaseTokenizer):
    """Placeholder for the trainer's tokenizer slot; vision inputs need none."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        pass

    def __init__(self, config: Config, **kwargs: Any) -> None:
        del config, kwargs
        super().__init__()

    def encode(self, *args, **kwargs) -> list[int]:
        raise NotImplementedError("Vision benchmark has no tokenizer.")

    def decode(self, *args, **kwargs) -> str:
        raise NotImplementedError("Vision benchmark has no tokenizer.")

    def get_vocab_size(self) -> int:
        return 0


class _HostToDeviceStats:
    """Process-local accumulator for timed host -> device copies."""

    num_bytes = 0
    seconds = 0.0
    num_copies = 0

    @classmethod
    def add(cls, num_bytes: int, seconds: float) -> None:
        cls.num_bytes += num_bytes
        cls.seconds += seconds
        cls.num_copies += 1

    @classmethod
    def pop(cls) -> tuple[int, float, int]:
        out = (cls.num_bytes, cls.seconds, cls.num_copies)
        cls.num_bytes, cls.seconds, cls.num_copies = 0, 0.0, 0
        return out


@dataclass(kw_only=True, slots=True)
class VolumeTrainingMicrobatch(TrainingMicrobatch):
    """One DP-rank-local microbatch of clean patches, ``(B * S, F)``."""

    labels: torch.Tensor
    loss_token_counts: torch.Tensor
    routing_token_counts: torch.Tensor
    draw_ids: list[int]
    """Globally unique id of each sample draw (epoch * dataset size + index);
    seeds the per-sample input noise so it does not depend on the layout."""
    measure_host_to_device: bool = False
    log_freq: int = 0

    def as_input_dict(self) -> dict[str, Any]:
        return {"labels": self.labels, "draw_ids": self.draw_ids}

    def to_input_dict(
        self, device: torch.device | str, *, non_blocking: bool = False
    ) -> dict[str, Any]:
        if not self.measure_host_to_device:
            labels = self.labels.to(device, non_blocking=non_blocking)
            return {"labels": labels, "draw_ids": self.draw_ids}
        device_utils.device_module.synchronize()
        start = time.perf_counter()
        labels = self.labels.to(device, non_blocking=non_blocking)
        device_utils.device_module.synchronize()
        _HostToDeviceStats.add(
            self.labels.numel() * self.labels.element_size(),
            time.perf_counter() - start,
        )
        if self.log_freq and _HostToDeviceStats.num_copies >= self.log_freq:
            num_bytes, seconds, num_copies = _HostToDeviceStats.pop()
            logger.info(
                f"host_to_device: {num_copies} copies, "
                f"{num_bytes / num_copies / 2**20:.1f} MiB/copy, "
                f"{seconds / num_copies * 1e3:.2f} ms/copy, "
                f"{num_bytes / seconds / 1e9:.2f} GB/s"
            )
        return {"labels": labels, "draw_ids": self.draw_ids}


def _collate(
    samples: list[tuple[torch.Tensor, int, float, float]],
) -> tuple[torch.Tensor, list[int], float, float]:
    """Stack samples; returns the summed read and preprocess seconds."""
    start = time.perf_counter()
    patches, draw_ids, read_seconds, preprocess_seconds = zip(*samples, strict=True)
    patches_TF = torch.cat(patches)
    stack_seconds = time.perf_counter() - start
    return (
        patches_TF,
        list(draw_ids),
        sum(read_seconds),
        sum(preprocess_seconds) + stack_seconds,
    )


class _VolumeDataset(Dataset):
    """Map-style dataset of patchified volumes; one item is ``(S, F)``."""

    def __init__(
        self,
        *,
        source: Literal["synthetic", "hdf5"],
        paths: list[str],
        dataset_key: str,
        num_samples: int,
        in_channels: int,
        volume_shape: tuple[int, int, int],
        patch_size: int,
        dtype: torch.dtype,
        seed: int,
        synthetic_pool_size: int = 0,
    ) -> None:
        self.source = source
        self.paths = paths
        self.dataset_key = dataset_key
        self.in_channels = in_channels
        self.volume_shape = volume_shape
        self.patch_size = patch_size
        self.dtype = dtype
        self.seed = seed
        self.synthetic_pool_size = synthetic_pool_size
        self._files: dict[str, Any] = {}
        self._pool: dict[int, torch.Tensor] = {}
        # (path, index within file) for every sample, built once on the host.
        self._index: list[tuple[str, int]] = []
        if source == "synthetic":
            self._num_samples = num_samples
            # Generated volumes are float32.
            self.bytes_per_sample = 4 * in_channels * math.prod(volume_shape)
            return
        import h5py

        self.bytes_per_sample = 0
        for path in paths:
            with h5py.File(path, "r") as f:
                data = f[dataset_key]
                self.bytes_per_sample = data.dtype.itemsize * math.prod(data.shape[1:])
                expected = (in_channels, *volume_shape)
                if tuple(data.shape[1:]) != expected:
                    raise ValueError(
                        f"{path}:{dataset_key} has sample shape "
                        f"{tuple(data.shape[1:])}, expected (C, D, H, W) = {expected}."
                    )
                self._index.extend((path, i) for i in range(data.shape[0]))
        self._num_samples = len(self._index)
        if self._num_samples == 0:
            raise ValueError(f"No samples found in {paths}.")

    def __len__(self) -> int:
        return self._num_samples

    def __getstate__(self) -> dict[str, Any]:
        # Open h5py handles cannot be pickled into worker processes.
        state = self.__dict__.copy()
        state["_files"] = {}
        state["_pool"] = {}
        return state

    def _generate_volume(self, idx: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(self.seed + idx)
        return torch.randn((self.in_channels, *self.volume_shape), generator=generator)

    def _read_volume(self, idx: int) -> torch.Tensor:
        if self.source == "synthetic":
            if self.synthetic_pool_size <= 0:
                return self._generate_volume(idx)
            key = idx % self.synthetic_pool_size
            volume = self._pool.get(key)
            if volume is None:
                volume = self._pool[key] = self._generate_volume(key)
            return volume
        import h5py

        path, local_idx = self._index[idx]
        # h5py handles are opened lazily per worker process; they are not
        # fork-safe to share.
        f = self._files.get(path)
        if f is None:
            f = self._files[path] = h5py.File(path, "r")
        return torch.from_numpy(f[self.dataset_key][local_idx])

    def __getitem__(
        self, key: tuple[int, int]
    ) -> tuple[torch.Tensor, int, float, float]:
        """Returns ``(patches, draw_id, read seconds, preprocess seconds)``."""
        idx, draw_id = key
        start = time.perf_counter()
        volume_CDHW = self._read_volume(idx)
        read_done = time.perf_counter()
        patches_SF = patchify(volume_CDHW.to(self.dtype), self.patch_size)
        patches_SF = patches_SF.contiguous()
        return patches_SF, draw_id, read_done - start, time.perf_counter() - read_done


class _DistributedInfiniteSampler:
    """Yield this DP rank's ``(index, draw_id)`` pairs forever, reshuffled
    every epoch.

    ``start`` skips samples already consumed, for checkpoint resumption.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        dp_rank: int,
        dp_world_size: int,
        shuffle: bool,
        seed: int,
        start: int,
    ) -> None:
        if num_samples < dp_world_size:
            raise ValueError(
                f"Dataset has {num_samples} samples, fewer than the "
                f"{dp_world_size} data-parallel ranks."
            )
        self.num_samples = num_samples
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.shuffle = shuffle
        self.seed = seed
        self.start = start

    def __iter__(self) -> Iterator[tuple[int, int]]:
        per_rank = self.num_samples // self.dp_world_size
        epoch, offset = divmod(self.start, per_rank)
        while True:
            if self.shuffle:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                order = torch.randperm(self.num_samples, generator=generator)
            else:
                order = torch.arange(self.num_samples)
            rank_order = order[self.dp_rank :: self.dp_world_size][:per_rank]
            for idx in rank_order[offset:].tolist():
                yield idx, epoch * self.num_samples + idx
            epoch, offset = epoch + 1, 0


class VolumeDataLoader(BaseDataLoader):
    """Multi-process volumetric loader with pinned, prefetched microbatches."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        volume_shape: tuple[int, int, int]
        patch_size: int
        in_channels: int = 1
        source: Literal["synthetic", "hdf5"] = "hdf5"
        """``hdf5`` reads ``(N, C, D, H, W)`` datasets from ``path_glob`` on
        disk; ``synthetic`` generates volumes in the workers instead."""
        path_glob: str = ""
        dataset_key: str = "volume"
        num_synthetic_samples: int = 4096
        """Index space of the synthetic source; every index is a distinct draw."""
        synthetic_pool_size: int = 0
        """0 regenerates every synthetic volume, so CPU RNG cost lands in the
        loader. N > 0 generates N volumes once per worker and cycles them, so a
        storage-free run measures only patchify, pinning and the copy."""
        loader_stats_log_freq: int = 10
        """Log the ``loader`` and ``data_rate`` lines every N microbatches;
        0 disables."""
        dtype: Literal["float32", "bfloat16"] = "float32"
        """Host and transfer dtype of the patches."""
        num_workers: int = 8
        prefetch_factor: int = 4
        pin_memory: bool = True
        shuffle: bool = True
        seed: int = 42
        measure_host_to_device: bool = True
        """Time each host -> device copy (adds two device syncs per microbatch)."""
        host_to_device_log_freq: int = 10
        dataloader_kwargs: dict[str, Any] = field(default_factory=dict)

        def __post_init__(self) -> None:
            BaseDataLoader.Config.__post_init__(self)
            if self.source == "hdf5" and not self.path_glob:
                raise ValueError(
                    "source='hdf5' requires path_glob (VIT_DATA_GLOB); "
                    "use source='synthetic' (VIT_DATA=synthetic) to run without data."
                )

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        max_context_length: int,
        num_tokens_per_microbatch: int,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if num_tokens_per_microbatch % max_context_length != 0:
            raise ValueError(
                f"num_tokens_per_microbatch ({num_tokens_per_microbatch}) must be a "
                f"multiple of tokens per sample ({max_context_length})."
            )
        self.config = config
        self.dp_world_size = dp_world_size
        self.samples_per_microbatch = num_tokens_per_microbatch // max_context_length
        self.num_samples_consumed = 0
        paths = sorted(glob.glob(config.path_glob)) if config.source == "hdf5" else []
        self.dataset = _VolumeDataset(
            source=config.source,
            paths=paths,
            dataset_key=config.dataset_key,
            num_samples=config.num_synthetic_samples,
            in_channels=config.in_channels,
            volume_shape=config.volume_shape,
            patch_size=config.patch_size,
            dtype=TORCH_DTYPE_MAP[config.dtype],
            seed=config.seed,
            synthetic_pool_size=config.synthetic_pool_size,
        )
        num_tokens_per_sample = self.dataset[(0, 0)][0].shape[0]
        if num_tokens_per_sample != max_context_length:
            raise ValueError(
                f"Samples have {num_tokens_per_sample} tokens but the model "
                f"expects {max_context_length}."
            )
        self.dp_rank = dp_rank
        self._log_host_memory(num_tokens_per_microbatch)
        self._iterator: Iterator[TrainingMicrobatch] | None = None
        self._loader: DataLoader | None = None

    def _log_host_memory(self, num_tokens_per_microbatch: int) -> None:
        """Log this rank's host buffer footprint, which bounds the volume size
        on nodes with little host memory."""
        config = self.config
        patch_dim = config.in_channels * config.patch_size**3
        microbatch_bytes = (
            num_tokens_per_microbatch
            * patch_dim
            * TORCH_DTYPE_MAP[config.dtype].itemsize
        )
        workers = max(1, config.num_workers)
        # Batches queued by the workers, plus the one being pinned and the one
        # in use; each worker also holds a raw volume while patchifying.
        in_flight = config.num_workers * config.prefetch_factor + 2
        total = in_flight * microbatch_bytes + workers * self.dataset.bytes_per_sample
        if config.source == "synthetic":
            total += workers * config.synthetic_pool_size * self.dataset.bytes_per_sample
        logger.info(
            f"data: {self.dataset.bytes_per_sample / 2**20:.1f} MiB read per sample, "
            f"{microbatch_bytes / 2**20:.1f} MiB per microbatch, "
            f"~{total / 2**30:.2f} GiB host buffers per rank"
        )

    def _build_iterator(self) -> Iterator[TrainingMicrobatch]:
        config = self.config
        sampler = _DistributedInfiniteSampler(
            num_samples=len(self.dataset),
            dp_rank=self.dp_rank,
            dp_world_size=self.dp_world_size,
            shuffle=config.shuffle,
            seed=config.seed,
            start=self.num_samples_consumed,
        )
        worker_kwargs: dict[str, Any] = {}
        if config.num_workers > 0:
            worker_kwargs = {
                "prefetch_factor": config.prefetch_factor,
                "persistent_workers": True,
            }
        self._loader = DataLoader(
            self.dataset,
            batch_size=self.samples_per_microbatch,
            sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory and device_utils.device_type != "cpu",
            collate_fn=_collate,
            drop_last=True,
            **worker_kwargs,
            **config.dataloader_kwargs,
        )
        log_freq = config.loader_stats_log_freq
        # The first window starts at the first delivery, so worker startup is
        # not counted.
        window_start: float | None = None
        window_microbatches, window_read, window_preprocess = 0, 0.0, 0.0
        for patches_TF, draw_ids, read_seconds, preprocess_seconds in self._loader:
            self.num_samples_consumed += self.samples_per_microbatch
            if window_start is None:
                window_start = time.perf_counter()
            else:
                window_microbatches += 1
                window_read += read_seconds
                window_preprocess += preprocess_seconds
            if log_freq and window_microbatches == log_freq:
                samples = window_microbatches * self.samples_per_microbatch
                num_bytes = samples * self.dataset.bytes_per_sample
                # Stages 1 and 2, timed inside the workers. Per-worker rates;
                # workers run concurrently. Synthetic volumes come from host
                # memory, so they get no read bandwidth.
                if config.source == "synthetic":
                    stage1 = f"generate {window_read / samples * 1e3:.2f} ms/sample"
                else:
                    stage1 = (
                        f"read {window_read / samples * 1e3:.2f} ms/sample "
                        f"({num_bytes / window_read / 1e9:.3f} GB/s per worker)"
                    )
                logger.info(
                    f"loader: {stage1}, "
                    f"preprocess {window_preprocess / samples * 1e3:.2f} ms/sample"
                )
                # Wall time between deliveries includes compute, so this is the
                # bandwidth training consumes, not the loader's peak rate.
                elapsed = time.perf_counter() - window_start
                logger.info(
                    f"data_rate: {samples / elapsed:.2f} samples/s, "
                    f"{num_bytes / elapsed / 1e9:.3f} GB/s read by this rank"
                )
                window_start = time.perf_counter()
                window_microbatches, window_read, window_preprocess = 0, 0.0, 0.0
            yield VolumeTrainingMicrobatch(
                labels=patches_TF,
                loss_token_counts=torch.tensor(patches_TF.numel(), dtype=torch.int64),
                routing_token_counts=torch.zeros(0, dtype=torch.int64),
                draw_ids=draw_ids,
                measure_host_to_device=config.measure_host_to_device,
                log_freq=config.host_to_device_log_freq,
            )

    def __iter__(self) -> Iterator[TrainingMicrobatch]:
        if self._iterator is None:
            self._iterator = self._build_iterator()
        return self._iterator

    def state_dict(self) -> dict[str, Any]:
        return {
            "dp_world_size": self.dp_world_size,
            f"dp_rank_{self.dp_rank}": self.num_samples_consumed,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict["dp_world_size"] != self.dp_world_size:
            raise ValueError(
                "cannot resume after changing the effective data-parallel degree"
            )
        self.num_samples_consumed = state_dict[f"dp_rank_{self.dp_rank}"]
        self._iterator = None
