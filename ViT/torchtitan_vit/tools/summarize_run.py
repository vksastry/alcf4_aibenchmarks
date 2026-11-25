"""Average the per-step metrics of one run and append them to a results CSV.

    python tools/summarize_run.py outputs/<run> [--warmup 5] [--csv results.csv]
    python tools/summarize_run.py --table results.csv

Reads ``train.log`` and ``run_info.txt`` written by ``scripts/aurora_run.sh``.
Steps up to ``--warmup`` (compilation, allocator and loader warm-up) are
dropped; the rest are averaged over steps and ranks:

- ``tps``, ``tflops``, ``mfu``: TorchTitan's per-device step metrics (mean and
  standard deviation). ``tflops_aggregate`` is the mean times the rank count.
- ``memory``: peak over steps and ranks.
- ``loader``, ``data_rate``, ``host_to_device``: the data-path lines, without
  the first window of each rank.

Writes ``summary.json`` next to the log and, with ``--csv``, appends one row.
Standard library only.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"(-?[\d,]*\.?\d+)"

STEP_FIELDS = {
    "step": re.compile(r"step:\s*" + NUMBER),
    "loss": re.compile(r"loss:\s*" + NUMBER),
    "memory_gib": re.compile(r"memory:\s*" + NUMBER + r"GiB"),
    "memory_pct": re.compile(r"memory:\s*[\d.]+GiB\(" + NUMBER + r"%\)"),
    "tps": re.compile(r"tps:\s*" + NUMBER),
    "tflops": re.compile(r"tflops:\s*" + NUMBER),
    "mfu": re.compile(r"mfu:\s*" + NUMBER + r"%"),
}
LOADER = re.compile(
    r"loader: (?:generate|read) " + NUMBER + r" ms/sample"
    r"(?: \(" + NUMBER + r" GB/s per worker\))?, preprocess " + NUMBER + r" ms/sample"
)
DATA_RATE = re.compile(r"data_rate: " + NUMBER + r" samples/s, " + NUMBER + r" GB/s")
H2D = re.compile(
    r"host_to_device: \d+ copies, [\d.]+ MiB/copy, " + NUMBER + r" ms/copy, "
    + NUMBER + r" GB/s"
)
ERROR = re.compile(r"(Error|error:|OutOfMemory|out of memory|Killed|BAD TERMINATION)")

COLUMNS = [
    "run", "status", "config", "nodes", "ranks", "flavor", "volume", "patch",
    "tokens_per_sample", "dp_replicate", "dp_shard", "tp", "cp", "cp_backend",
    "pp", "pp_microbatches", "samples_per_microbatch", "remat", "data",
    "steps_averaged", "tps_mean", "tps_std", "tflops_mean", "tflops_std",
    "tflops_aggregate", "mfu_mean", "memory_peak_gib", "memory_peak_pct",
    "loss_first", "loss_last", "loader_stage1_ms", "loader_read_gbps_per_worker",
    "preprocess_ms", "data_rate_samples_s", "data_rate_gbps", "h2d_ms", "h2d_gbps",
    "error",
]
TABLE_COLUMNS = [
    "run", "status", "flavor", "tokens_per_sample", "tp", "cp", "pp",
    "dp_replicate", "dp_shard", "tps_mean", "tflops_mean", "tflops_aggregate",
    "memory_peak_gib", "data_rate_samples_s", "h2d_gbps",
]


def _num(text):
    return float(text.replace(",", ""))


def _mean(values):
    return statistics.mean(values) if values else None


def _std(values):
    return statistics.stdev(values) if len(values) > 1 else None


def read_run_info(run_dir):
    info, env = {}, {}
    path = os.path.join(run_dir, "run_info.txt")
    if not os.path.exists(path):
        return info, env
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("VIT_") and "=" in line:
                key, value = line.split("=", 1)
                env[key] = value
            else:
                for item in line.split():
                    if "=" in item:
                        key, value = item.split("=", 1)
                        info[key] = value
    return info, env


def tokens_per_sample(volume, patch):
    edges = [int(e) for e in volume.lower().split("x")]
    edges = edges * 3 if len(edges) == 1 else edges
    return math.prod(e // patch for e in edges)


def summarize(run_dir, warmup):
    info, env = read_run_info(run_dir)
    ranks = int(info.get("ranks", 1))
    volume = env.get("VIT_VOLUME", "256")
    patch = int(env.get("VIT_PATCH", 8))
    pp = env.get("VIT_PP", "1")
    row = {
        "run": os.path.basename(os.path.normpath(run_dir)),
        "config": info.get("config", ""),
        "nodes": info.get("nodes", ""),
        "ranks": ranks,
        "flavor": env.get("VIT_FLAVOR", "large"),
        "volume": volume,
        "patch": patch,
        "tokens_per_sample": tokens_per_sample(volume, patch),
        "dp_replicate": env.get("VIT_DP_REPLICATE", "1"),
        "dp_shard": env.get("VIT_DP_SHARD", "-1"),
        "tp": env.get("VIT_TP", "1"),
        "cp": env.get("VIT_CP", "1"),
        "cp_backend": env.get("VIT_CP_BACKEND", "ulysses"),
        "pp": pp,
        "pp_microbatches": env.get("VIT_PP_MICROBATCHES", pp),
        "samples_per_microbatch": env.get("VIT_SAMPLES_PER_MICROBATCH", "1"),
        "remat": env.get("VIT_REMAT", "default"),
        "data": env.get("VIT_DATA", "hdf5"),
    }

    steps = {key: [] for key in STEP_FIELDS}
    losses, loader, data_rate, h2d = [], [], [], []
    last_error = ""
    log_path = os.path.join(run_dir, "train.log")
    if os.path.exists(log_path):
        with open(log_path, errors="replace") as f:
            for raw in f:
                line = ANSI.sub("", raw)
                if "step:" in line and "tps:" in line:
                    fields = {}
                    for key, pattern in STEP_FIELDS.items():
                        match = pattern.search(line)
                        if match:
                            fields[key] = _num(match.group(1))
                    if "step" not in fields or "tps" not in fields:
                        continue
                    if "loss" in fields:
                        losses.append((fields["step"], fields["loss"]))
                    if fields["step"] <= warmup:
                        continue
                    for key, value in fields.items():
                        steps[key].append(value)
                    continue
                match = LOADER.search(line)
                if match:
                    loader.append(match.groups())
                    continue
                match = DATA_RATE.search(line)
                if match:
                    data_rate.append(match.groups())
                    continue
                match = H2D.search(line)
                if match:
                    h2d.append(match.groups())
                    continue
                if ERROR.search(line):
                    last_error = line.strip()[-300:]

    # The first window of every rank includes warm-up; ranks interleave in
    # the log, so drop the first ``ranks`` lines of each data-path metric.
    loader, data_rate, h2d = loader[ranks:], data_rate[ranks:], h2d[ranks:]

    row["steps_averaged"] = len({int(s) for s in steps["step"]})
    row["status"] = "OK" if steps["tps"] else "FAILED"
    row["error"] = "" if steps["tps"] else last_error
    row["tps_mean"] = _mean(steps["tps"])
    row["tps_std"] = _std(steps["tps"])
    row["tflops_mean"] = _mean(steps["tflops"])
    row["tflops_std"] = _std(steps["tflops"])
    if row["tflops_mean"] is not None:
        row["tflops_aggregate"] = row["tflops_mean"] * ranks
    row["mfu_mean"] = _mean(steps["mfu"])
    row["memory_peak_gib"] = max(steps["memory_gib"], default=None)
    row["memory_peak_pct"] = max(steps["memory_pct"], default=None)
    if losses:
        losses.sort()
        row["loss_first"], row["loss_last"] = losses[0][1], losses[-1][1]
    row["loader_stage1_ms"] = _mean([_num(g[0]) for g in loader])
    row["loader_read_gbps_per_worker"] = _mean([_num(g[1]) for g in loader if g[1]])
    row["preprocess_ms"] = _mean([_num(g[2]) for g in loader])
    row["data_rate_samples_s"] = _mean([_num(g[0]) for g in data_rate])
    row["data_rate_gbps"] = _mean([_num(g[1]) for g in data_rate])
    row["h2d_ms"] = _mean([_num(g[0]) for g in h2d])
    row["h2d_gbps"] = _mean([_num(g[1]) for g in h2d])
    return row


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}" if abs(value) < 1e4 else f"{value:.0f}"
    return str(value)


def print_table(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f, restval=""))
    if not rows:
        return
    width = {c: max(len(c), *(len(r.get(c, "")) for r in rows)) for c in TABLE_COLUMNS}
    print("  ".join(c.ljust(width[c]) for c in TABLE_COLUMNS))
    for r in rows:
        print("  ".join(r.get(c, "").ljust(width[c]) for c in TABLE_COLUMNS))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="?", help="Run output directory")
    parser.add_argument("--warmup", type=int, default=5, help="Steps to drop")
    parser.add_argument("--csv", help="Append the summary as a row of this CSV")
    parser.add_argument("--table", metavar="CSV", help="Print a results CSV and exit")
    args = parser.parse_args()
    if args.table:
        print_table(args.table)
        return
    if not args.run_dir:
        parser.error("run_dir is required")

    row = summarize(args.run_dir, args.warmup)
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(row, f, indent=2)
    if args.csv:
        new = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new:
                writer.writeheader()
            writer.writerow({k: _fmt(row.get(k)) for k in COLUMNS})

    if row["status"] == "OK":
        print(
            f"summary ({row['steps_averaged']} steps after {args.warmup} warm-up, "
            f"{row['ranks']} ranks): tps {_fmt(row['tps_mean'])} "
            f"+- {_fmt(row['tps_std'])}, tflops/device {_fmt(row['tflops_mean'])}, "
            f"tflops total {_fmt(row['tflops_aggregate'])}, "
            f"peak memory {_fmt(row['memory_peak_gib'])} GiB, "
            f"data_rate {_fmt(row['data_rate_samples_s'])} samples/s, "
            f"h2d {_fmt(row['h2d_gbps'])} GB/s"
        )
    else:
        print(f"summary: FAILED, no steps after warm-up. {row['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
