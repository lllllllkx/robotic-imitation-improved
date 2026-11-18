# coding=utf-8
"""Parse PPO console.log into a per-iteration training-curve CSV/JSON.

Includes the curriculum episode limit implied by the aggregate step count:
  t = clip(steps / curriculum_steps, 0, 1); limit = 20 + (600 - 20) * t**3
This is the *scheduled* limit, not the observed episode length.
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import io
import json
import re

BLOCK_RE = re.compile(r"\*\*\*\*\*\*\*\*\*\* Iteration (\d+) \*\*\*\*\*\*\*\*\*\*")
FIELDS = {
    "ep_len_mean": r"\|\s*EpLenMean\s*\|\s*([0-9.eE+-]+)",
    "ep_rew_mean": r"\|\s*EpRewMean\s*\|\s*([0-9.eE+-]+)",
    "episodes_so_far": r"\|\s*EpisodesSoFar\s*\|\s*([0-9]+)",
    "time_elapsed": r"\|\s*TimeElapsed\s*\|\s*([0-9.]+(?:e[+-]?[0-9]+)?)",
    "timesteps_so_far": r"\|\s*TimestepsSoFar\s*\|\s*([0-9]+)",
}


def parse(path):
    text = io.open(path, encoding="utf-8", errors="replace").read()
    parts = BLOCK_RE.split(text)
    rows = []
    for index in range(1, len(parts), 2):
        iteration = int(parts[index])
        body = parts[index + 1]
        row = {"iteration": iteration}
        for key, pattern in FIELDS.items():
            match = re.search(pattern, body)
            if not match:
                row[key] = None
            elif key in ("episodes_so_far", "timesteps_so_far"):
                row[key] = int(match.group(1))
            else:
                row[key] = float(match.group(1))
        rows.append(row)
    previous_time = None
    for row in rows:
        if row["time_elapsed"] is not None and previous_time is not None:
            row["iteration_seconds"] = row["time_elapsed"] - previous_time
        else:
            row["iteration_seconds"] = None
        previous_time = row["time_elapsed"]
    return rows


def curriculum_limit(steps, total_steps, start=20, end=600):
    if steps is None:
        return None
    t = max(0.0, min(1.0, float(steps) / float(total_steps)))
    return int((1.0 - t ** 3) * start + (t ** 3) * end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--curriculum-steps", type=int, default=30000000)
    ap.add_argument("--resume-step-count", type=int, default=0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    rows = [r for r in parse(args.log) if r["timesteps_so_far"] is not None]
    for row in rows:
        row["total_timesteps_so_far"] = (
            args.resume_step_count + row["timesteps_so_far"])
        row["curriculum_limit"] = curriculum_limit(
            row["total_timesteps_so_far"], args.curriculum_steps)

    with io.open(args.out_prefix + ".csv", "w", newline="",
                 encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteration", "timesteps_so_far", "ep_len_mean",
                         "ep_rew_mean", "episodes_so_far", "time_elapsed",
                         "iteration_seconds", "total_timesteps_so_far",
                         "curriculum_limit"])
        for row in rows:
            writer.writerow([row["iteration"], row["timesteps_so_far"],
                             row["ep_len_mean"], row["ep_rew_mean"],
                             row["episodes_so_far"], row["time_elapsed"],
                             row["iteration_seconds"],
                             row["total_timesteps_so_far"],
                             row["curriculum_limit"]])
    with io.open(args.out_prefix + ".json", "w", encoding="utf-8",
                 newline="\n") as fh:
        json.dump({"log": args.log, "curriculum_steps": args.curriculum_steps,
                   "rows": rows}, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print("rows=%d last_steps=%s last_ep_len=%s last_limit=%s"
          % (len(rows), rows[-1]["timesteps_so_far"] if rows else None,
             rows[-1]["ep_len_mean"] if rows else None,
             rows[-1]["curriculum_limit"] if rows else None))


if __name__ == "__main__":
    main()
