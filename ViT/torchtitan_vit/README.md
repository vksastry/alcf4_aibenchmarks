# 3D ViT benchmark on TorchTitan

A 3D Vision Transformer encoder for the ALCF-4 AI benchmark suite. It is
built on [TorchTitan](https://github.com/pytorch/torchtitan) as an external
package and does not modify TorchTitan. It replaces the Megatron-DeepSpeed
implementation in `../Megatron-DeepSpeed_ViT`.

| Item | Value |
|---|---|
| TorchTitan | `main` at `ff96670a26c7079bc1b9049f5b6942adf35567c1` (2026-10-30). TorchTitan makes no API compatibility promises, so keep this pin. |
| PyTorch | nightly (developed against `2.16.0.dev20261030`) |
| Model | patch embedding -> N pre-norm blocks (MHA + SwiGLU, RMSNorm, 1D RoPE) -> per-patch head |
| Attention | **bidirectional (non-causal)** `scaled_dot_product_attention` |
| Objective | denoising reconstruction: the model regresses clean patches from noisy ones (MSE) |

## Layout

```
vit3d/attention.py   non-causal SDPA, plus its Ulysses and all-gather-K/V CP variants
vit3d/model.py       ViT3DModel (a TorchTitan Decoder subclass), sharding contracts, FLOP count
vit3d/flavors.py     model sizes and the sharded config builder
vit3d/data.py        volumetric loader: HDF5/synthetic -> patchify -> pinned memory -> device
vit3d/recipes.py     vit3d_scaling (benchmark entry point) and vit3d_debug_* test recipes
scripts/aurora_vit3d.pbs   PBS template for Aurora
tools/make_synthetic_h5.py writes HDF5 shards for the I/O path
tests/                     CPU/gloo harness and the parallelism equivalence matrix
```

## Scaling dimensions and the communication each one exercises

All degrees multiply to the world size. `VIT_DP_SHARD=-1` takes whatever ranks remain.

| Axis | Env var | Splits | Collective | Network tier |
|---|---|---|---|---|
| DDP / HSDP replicate | `VIT_DP_REPLICATE` | batch | gradient all-reduce | scale-out |
| FSDP2 | `VIT_DP_SHARD` | params, grads, optimizer state | param all-gather, grad reduce-scatter | scale-out / scale-up |
| TP + sequence parallel | `VIT_TP`, `VIT_SP` | weights and heads | activation all-gather and reduce-scatter on the critical path | scale-up |
| PP | `VIT_PP`, `VIT_PP_SCHEDULE`, `VIT_PP_MICROBATCHES` | layers | point-to-point send/recv | latency |
| CP, Ulysses | `VIT_CP`, `VIT_CP_BACKEND=ulysses` | sequence <-> heads | 4 all-to-alls per layer forward, 4 backward | bisection bandwidth |
| CP, all-gather K/V | `VIT_CP`, `VIT_CP_BACKEND=allgather_kv` | sequence | K/V all-gather, dK/dV reduce-scatter | scale-out |

Constraints:
- Ulysses needs `n_heads % (tp * cp) == 0`. The `large_h48` flavor (48 heads)
  lets Ulysses span a 12-tile Aurora node.
- All-gather K/V needs one sample per microbatch per CP group.
- TorchTitan has a single `cp` mesh axis, so Ulysses and all-gather K/V cannot be
  combined (Ulysses inside a node, all-gather across nodes) without changing
  TorchTitan core. TorchTitan `main` also has no ring attention.

## Problem size and figure of merit

`tokens S = D*H*W / p^3`. The configurable inputs are `VIT_VOLUME` (cube edge
or `DxHxW`), `VIT_PATCH`, `VIT_CHANNELS`, `VIT_FLAVOR` (debugmodel, base,
large, huge, giant, large_h48) and `VIT_SAMPLES_PER_MICROBATCH`.

| Volume | p | S |
|---|---|---|
| 256^3 | 8 | 32,768 |
| 512^3 | 8 | 262,144 |
| 1024^3 | 16 | 262,144 |
| 1024^3 | 8 | 2,097,152 |

TorchTitan logs `tflops` per device as **model FLOPs**:
`6 * params + sum over layers of 6 * H * (2 * head_dim) * S` per token. This
count has no causal discount (the attention is non-causal) and does not count
recomputation. It is 12x the README FOM `b_s L (N^2 d + 6 d^2 N) / T`, plus
the small patch-embedding and head terms. Report it together with:

- `data_loading(%)` (in TensorBoard metrics): the share of step time spent waiting on the loader.
- `host_to_device` log lines (`VIT_MEASURE_H2D=1`): achieved host-to-device GB/s per microbatch copy.
- `loader` and `data_rate` log lines: disk read and CPU preprocessing time, and consumed read bandwidth (see Data path).

LLM benchmark note: TorchTitan's FLOP convention does **not** halve attention
FLOPs for causal models. Halve them there when comparing against this
benchmark.

## Data path

Every step runs the whole pipeline, from the disk read to the model's compute
and communication: HDF5 on disk -> loader workers read the volume -> CPU
patchify and microbatch stacking -> pinned host memory -> host-to-device copy
-> model.

What is measured (logged per rank every 10 microbatches; ignore the first
lines, which include warm-up):

| Log line | Stage | Reports |
|---|---|---|
| `loader` | 1. disk read, timed in the workers | ms per sample and GB/s per worker (workers run concurrently) |
| `loader` | 2. CPU preprocessing (patchify + stacking), timed in the workers | ms per sample |
| `data_rate` | whole loader, as consumed by training | samples/s and GB/s read by this rank |
| `host_to_device` | 3. host-to-device copy (`VIT_MEASURE_H2D=1`, default) | ms and GB/s per copy |
| `data_loading(%)` (TorchTitan) | wait on the loader | share of step time |

At startup the loader also logs bytes per sample and this rank's host buffer
footprint. Standalone storage characterization (IOR, h5bench) is a separate
benchmark.

Data sources:

- `VIT_DATA=hdf5` (default) with `VIT_DATA_GLOB` reads `(N, C, D, H, W)`
  datasets named `volume` (override with `VIT_DATA_KEY`). Generate data with
  `tools/make_synthetic_h5.py`. If the dataset fits in host memory, repeated
  reads come from the page cache and stage 1 measures memory, not disk; make
  it larger than the aggregate host memory where storage allows.
- `VIT_DATA=synthetic` runs without any data: the workers generate volumes in
  host memory, and stage 1 is reported as `generate`. With
  `VIT_SYNTHETIC_POOL=4` (default) each worker generates 4 volumes once and
  cycles them; `VIT_SYNTHETIC_POOL=0` regenerates every volume.
- `VIT_NUM_WORKERS` (default 8) and `VIT_PREFETCH` (default 4) set loader
  workers per rank and batches queued per worker.
  `VIT_DATA_DTYPE=bfloat16` halves host-to-device bytes.
- Known amplification: every CP and TP rank of a DP replica reads the full
  sample and slices it on the device. Per-rank slab reads are not implemented yet.

## Running

```bash
# Environment (once)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/<cu13x|xpu|rocm>
git clone https://github.com/pytorch/torchtitan && cd torchtitan
git checkout ff96670a26c7079bc1b9049f5b6942adf35567c1
pip install -r requirements.txt && pip install -e .

# Single node, 12 ranks: Ulysses CP over the node at 256^3
cd alcf4_aibenchmarks/ViT/torchtitan_vit
PYTHONPATH=$PWD VIT_FLAVOR=large_h48 VIT_VOLUME=256 VIT_CP=12 VIT_DATA_GLOB='/path/*.h5' \
  torchrun --nproc_per_node 12 -m torchtitan.train \
  --module vit3d.recipes --config vit3d_scaling

# Aurora
qsub -l select=8 -v VIT_VENV=/path/venv,VIT_TP=2,VIT_CP=6,VIT_VOLUME=512 scripts/aurora_vit3d.pbs
```

## Validation

`tests/run_cpu_matrix.sh` runs every parallelism on CPU with gloo, using a
patched device shim (`tests/cpu_train.py`). Each run loads one seed
checkpoint, and every run must reproduce the single-rank loss curve.
Per-sample noise is seeded by sample id, so the inputs do not depend on the
parallel layout.

```bash
PYTHON=/path/to/venv/bin/python tests/run_cpu_matrix.sh            # all cases
PYTHON=/path/to/venv/bin/python tests/run_cpu_matrix.sh tp cp_ulysses
```

Status on 2026-10-30, all on CPU/gloo:

| Case | Ranks | Result |
|---|---|---|
| FSDP, DDP, TP+SP, PP (1F1B), CP Ulysses, FSDP x CP x TP | 1-8 | losses and grad norms match the single-rank run to 5 digits for all 10 steps |
| CP all-gather K/V | 2 | matches the one-sample-per-microbatch baseline |
| HDF5 loader, 2 workers, H2D timing, bf16 FSDP | 2 | trains; timing lines are logged |

**Not yet validated:** any GPU/XPU run, HSDP, interleaved PP schedules,
performance, and the Aurora job script. The SDPA path on XPU should use the
SYCL-TLA flash kernel; confirm that with a profile on the first Aurora run.
