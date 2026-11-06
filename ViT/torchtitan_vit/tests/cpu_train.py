"""Run torchtitan.train on CPU/gloo by forcing the device type before imports.

Local functional testing only (no accelerator on this machine).
"""
import types

import torch
import torch.distributed as dist

import torchtitan.tools.utils as tu

_cpu = types.SimpleNamespace(
    **{k: getattr(torch.cpu, k) for k in dir(torch.cpu) if not k.startswith("__")}
)
_cpu.current_device = lambda: 0
_cpu.set_device = lambda *a, **k: None
_cpu.device_count = lambda: 64
_cpu.empty_cache = lambda: None
_cpu.reset_peak_memory_stats = lambda *a, **k: None
_cpu.get_device_name = lambda *a, **k: "cpu"
_cpu.get_device_properties = lambda *a, **k: types.SimpleNamespace(
    name="cpu", total_memory=64 * 2**30, max_compute_units=1
)
_cpu.memory_stats = lambda *a, **k: {
    "active_bytes.all.peak": 0,
    "reserved_bytes.all.peak": 0,
    "num_alloc_retries": 0,
    "num_ooms": 0,
}
_cpu.is_initialized = lambda: True
_cpu.memory = types.SimpleNamespace(
    _record_memory_history=lambda *a, **k: None,
    _dump_snapshot=lambda *a, **k: None,
)
tu.device_type = "cpu"
tu.device_module = _cpu

_barrier = dist.barrier


def _cpu_barrier(*args, device_ids=None, **kwargs):
    return _barrier(*args, **kwargs)


dist.barrier = _cpu_barrier

# Gloo has no PREMUL_SUM; force FSDP2 to divide gradients outside the
# reduce-scatter (same result, separate divide kernel).
import torch.distributed.fsdp._fully_shard._fsdp_collectives as _fsdp_collectives  # noqa: E402

_get_factors = _fsdp_collectives._get_gradient_divide_factors


def _sum_only_factors(
    reduce_scatter_group,
    all_reduce_group,
    reduce_dtype,
    device_type="",
    factor=None,
    force_sum_reduction_for_comms=False,
):
    return _get_factors(
        reduce_scatter_group,
        all_reduce_group,
        reduce_dtype,
        device_type,
        factor,
        force_sum_reduction_for_comms=True,
    )


_fsdp_collectives._get_gradient_divide_factors = _sum_only_factors

from torchtitan.train import main  # noqa: E402

# Guarded: macOS spawns DataLoader workers by re-importing this script.
if __name__ == "__main__":
    main()
