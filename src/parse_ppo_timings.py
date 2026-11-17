# coding=utf-8
"""Summarise PPO steady-state throughput from console.log TimeElapsed rows.

Steady state = iterations 2..N (the first iteration carries TF graph warmup),
excluding any iteration whose elapsed delta is negative or exceeds
``--spike-factor`` times the median positive delta. Excluded runs keep their
raw data and a recorded reason; they are never averaged in.
"""

from __future__ import absolute_import, division, print_function

import argparse
import io
import json
import os
import re

ELAPSED_RE = re.compile(r"\|\s*TimeElapsed\s*\|\s*([0-9.]+)")
STEPS_RE = re.compile(r"\|\s*TimestepsSoFar\s*\|\s*([0-9]+)")


def parse_log(path, spike_factor=3.0, degradation_factor=1.5):
    text = io.open(path, encoding="utf-8", errors="replace").read()
    elapsed = [float(x) for x in ELAPSED_RE.findall(text)]
    steps = [int(x) for x in STEPS_RE.findall(text)]
    rows = []
    for i in range(1, min(len(elapsed), len(steps))):
        dt = elapsed[i] - elapsed[i - 1]
        ds = steps[i] - steps[i - 1]
        rows.append({"iteration": i, "steps": ds, "elapsed_s": dt,
                     "steps_per_s": (ds / dt if dt > 0 else None)})
    positive = sorted(r["elapsed_s"] for r in rows if r["elapsed_s"] > 0)
    median = positive[len(positive) // 2] if positive else 0.0
    for r in rows:
        r["anomaly"] = (r["elapsed_s"] <= 0 or
                        r["elapsed_s"] > spike_factor * median)
    valid = [r for r in rows if not r["anomaly"]]
    steady_steps = sum(r["steps"] for r in valid)
    steady_time = sum(r["elapsed_s"] for r in valid)
    monotonic = all(r["elapsed_s"] > 0 for r in rows)
    max_ratio = (max((r["elapsed_s"] for r in valid), default=0.0) / median
                 if median > 0 else None)
    reasons = []
    if not monotonic:
        reasons.append("non_monotonic_elapsed")
    if max_ratio is not None and max_ratio > degradation_factor:
        reasons.append("steady_iteration_degradation")
    return {
        "iterations": [{"iteration": 0}],
        "steady_rows": rows,
        "valid_iterations": len(valid),
        "total_iterations": len(rows),
        "steady_steps": steady_steps,
        "steady_time_s": steady_time,
        "steady_steps_per_second": (steady_steps / steady_time
                                    if steady_time > 0 else None),
        "median_iteration_s": median,
        "max_steady_iteration_ratio": max_ratio,
        "monotonic_elapsed": monotonic,
        "invalid_reasons": reasons,
        "excluded_iterations": [r["iteration"] for r in rows if r["anomaly"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("NAME", "PATH"),
                    required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spike-factor", type=float, default=3.0)
    ap.add_argument("--degradation-factor", type=float, default=1.5)
    args = ap.parse_args()

    runs = []
    for name, path in args.run:
        summary_path = os.path.join(os.path.dirname(path), "training_summary.json")
        wall = None
        if os.path.isfile(summary_path):
            data = json.load(io.open(summary_path, encoding="utf-8"))
            wall = data.get("wall_time_seconds")
        result = parse_log(path, args.spike_factor,
                           args.degradation_factor)
        result.update({"name": name, "console_log": os.path.abspath(path),
                       "wall_time_s": wall})
        result["valid"] = (bool(result["steady_steps_per_second"]) and
                           not result["invalid_reasons"])
        runs.append(result)

    valid = [r for r in runs if r["valid"]]
    pooled_steps = sum(r["steady_steps"] for r in valid)
    pooled_time = sum(r["steady_time_s"] for r in valid)
    payload = {
        "measurement": "PPO end-to-end steady-state, 600-step episodes, "
                       "pretrained init, randomizer off, 8 MPI processes",
        "spike_factor": args.spike_factor,
        "runs": runs,
        "valid_runs": [r["name"] for r in valid],
        "pooled": {
            "steps": pooled_steps,
            "time_s": pooled_time,
            "steps_per_second": (pooled_steps / pooled_time
                                 if pooled_time > 0 else None),
        },
        "per_run_steps_per_second_mean": (
            sum(r["steady_steps_per_second"] for r in valid) / len(valid)
            if valid else None),
        "per_run_steps_per_second_min": min(
            (r["steady_steps_per_second"] for r in valid), default=None),
        "per_run_steps_per_second_max": max(
            (r["steady_steps_per_second"] for r in valid), default=None),
    }
    with io.open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(json.dumps(payload["pooled"], ensure_ascii=False))


if __name__ == "__main__":
    main()
