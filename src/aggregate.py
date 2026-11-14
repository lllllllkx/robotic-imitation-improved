"""Collect ``results/eval/*/summary.json`` into one comparison table.

Outputs
-------
results/summary_table.csv   one row per (skill, variant)
results/summary_table.md    markdown table used in README / docs

Isolation: every summary records the ``--run-id`` it was produced with (and
the policy SHA256). Passing ``--run-id`` restricts the table to that run, so
stale directories left behind by earlier experiments cannot leak into a
comparison. Without it the table covers everything on disk and the distinct
run ids are printed, which makes an accidental mix visible instead of silent.
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import csv
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


COLUMNS = [
    ("skill", lambda p: p["tag"]),
    ("variant", lambda p: p["variant"]),
    ("init", lambda p: p.get("protocol", {}).get("init_mode", "controlled")),
    ("episodes", lambda p: str(p["aggregate"]["num_episodes"]),
     ),
    ("mean_reward", lambda p: "%.4f" % p["aggregate"]["mean_reward_mean"]),
    ("std_reward", lambda p: "%.4f" % p["aggregate"]["mean_reward_std"]),
    ("seed_mean", lambda p: "%.4f" % p["by_seed"]["mean_reward_mean_across_seeds"]),
    ("seed_std", lambda p: "%.4f" % p["by_seed"]["mean_reward_std_across_seeds"]),
    ("pose_err_rad", lambda p: "%.4f" % p["aggregate"]["mean_pose_err_rad_mean"]),
    ("root_err_m", lambda p: "%.4f" % p["aggregate"]["mean_root_err_m_mean"]),
    ("yaw_err_rad", lambda p: "%.4f" % p["aggregate"]["mean_yaw_err_rad_mean"]),
    ("yaw_drift_rad_s",
     lambda p: "%.4f" % p["aggregate"]["yaw_drift_rad_per_s_mean"]),
    ("speed_mps", lambda p: "%.4f" % p["aggregate"]["mean_speed_mps_mean"]),
    ("survival_pct",
     lambda p: "%.0f" % (100.0 * sum(
         1 for e in p["aggregate"]["episodes"]
         if e.get("termination", "time_limit") == "time_limit")
         / max(1, p["aggregate"]["num_episodes"]))),
    ("sha8", lambda p: (p.get("model_sha256") or "")[:8]),
]


def load(path):
  with open(path) as fh:
    return json.load(fh)


def collect(results_dir, run_id=None, verbose=True):
  rows = []
  skipped = []
  all_ids = set()
  for path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
    payload = load(path)
    all_ids.add(payload.get("run_id"))
    if run_id is not None and payload.get("run_id") != run_id:
      skipped.append(os.path.basename(os.path.dirname(path)))
      continue
    rows.append({name: fn(payload) for name, fn in COLUMNS})
  if verbose and run_id is None and len(all_ids) > 1:
    print("warning: summaries come from several run ids: %s"
          % sorted(str(i) for i in all_ids))
    print("         pass --run-id <id> to build a single-run table")
  if verbose and skipped:
    print("note: %d directories excluded by --run-id %s"
          % (len(skipped), run_id))
  return rows


def main(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument("--results-dir", default=os.path.join(REPO_ROOT, "results", "eval"))
  p.add_argument("--run-id", default=None,
                 help="only include summaries produced with this run id")
  p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
  args = p.parse_args(argv)

  rows = collect(args.results_dir, args.run_id)
  if not rows:
    raise SystemExit("no summaries in %s (run-id filter: %s)"
                     % (args.results_dir, args.run_id))

  csv_path = os.path.join(args.out_dir, "summary_table.csv")
  with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=[c[0] for c in COLUMNS])
    writer.writeheader()
    writer.writerows(rows)

  header = [c[0] for c in COLUMNS]
  lines = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
  for row in rows:
    lines.append("| " + " | ".join(row[h] for h in header) + " |")
  md_path = os.path.join(args.out_dir, "summary_table.md")
  with open(md_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")

  print("\n".join(lines))
  print("\n-> %s\n-> %s" % (csv_path, md_path))


if __name__ == "__main__":
  main()
