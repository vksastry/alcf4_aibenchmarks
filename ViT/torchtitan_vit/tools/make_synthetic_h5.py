"""Write synthetic volumetric HDF5 shards for the I/O path of the benchmark.

Each shard holds a ``(N, C, D, H, W)`` float32 dataset named ``volume``.
Size the total above the aggregate host memory of the nodes under test so
reads come from the filesystem, not the page cache.

    python tools/make_synthetic_h5.py --out /lus/flare/.../vit3d --shards 64 \
        --samples-per-shard 16 --volume 256 --channels 1
"""

import argparse
import os

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--samples-per-shard", type=int, default=8)
    parser.add_argument(
        "--volume", default="64", help="Cube edge, or DxHxW, in voxels"
    )
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument(
        "--chunk-per-sample",
        action="store_true",
        help="One HDF5 chunk per sample (default: contiguous layout)",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    edges = [int(e) for e in args.volume.lower().split("x")]
    shape = tuple(edges * 3 if len(edges) == 1 else edges)
    sample_shape = (args.channels, *shape)
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    sample_bytes = int(np.prod(sample_shape)) * 4
    for shard in range(args.shards):
        path = os.path.join(args.out, f"vit3d_{shard:05d}.h5")
        with h5py.File(path, "w") as f:
            dataset = f.create_dataset(
                "volume",
                shape=(args.samples_per_shard, *sample_shape),
                dtype="float32",
                chunks=(1, *sample_shape) if args.chunk_per_sample else None,
            )
            for i in range(args.samples_per_shard):
                dataset[i] = rng.standard_normal(sample_shape, dtype=np.float32)
        print(f"wrote {path}")
    total = args.shards * args.samples_per_shard * sample_bytes
    print(
        f"{args.shards * args.samples_per_shard} samples of {sample_shape}, "
        f"{sample_bytes / 2**20:.1f} MiB each, {total / 2**30:.2f} GiB total"
    )


if __name__ == "__main__":
    main()
