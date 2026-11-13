"""Collect every ``results/eval/*/summary.json`` into one comparison table.

Outputs
-------
results/summary_table.csv   one row per (skill, variant)
results/summary_table.md    markdown table used in README / docs
"""

from __future__ import absolute_import
from __future__ import division

import csv
import glob
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLUMNS = [
    ("skill", lambda p: p["tag"]),
    ("variant", lambda p: p["variant"]),
    ("episodes", lambda p: str(p["aggregate"]["num_episodes"])),
    ("mean_reward", lambda p: "%.4f" % p["aggregate"]["mean_reward_mean"]),
    ("std_reward", lambda p: "%.4f" % p["aggregate"]["mean_reward_std"]),
    ("pose_err_rad", lambda p: "%.4f" % p["aggregate"]["mean_pose_err_rad_mean"]),
    ("root_err_m", lambda p: "%.4f" % p["aggregate"]["mean_root_err_m_mean"]),
    ("yaw_err_rad", lambda p: "%.4f" % p["aggregate"]["mean_yaw_err_rad_mean"]),
    ("yaw_drift_rad_s",
     lambda p: "%.4f" % p["aggregate"]["yaw_drift_rad_per_s_mean"]),
    ("speed_mps", lambda p: "%.4f" % p["aggregate"]["mean_speed_mps_mean"]),
    ("survival_pct",
     lambda p: "%.0f" % (100.0 * sum(
         1 for e in p["aggregate"]["episodes"] if e["survived"])
         / max(1, p["aggregate"]["num_episodes"]))),
]


def collect(results_dir):
  rows = []
  for path in sorted(glob.glob(os.path.join(results_dir, "*", "summary.json"))):
    with open(path) as fh:
      payload = json.load(fh)
    rows.append({name: fn(payload) for name, fn in COLUMNS})
  return rows


def main():
  results_dir = os.path.join(REPO_ROOT, "results", "eval")
  rows = collect(results_dir)
  if not rows:
    raise SystemExit("no summaries under %s" % results_dir)

  csv_path = os.path.join(REPO_ROOT, "results", "summary_table.csv")
  with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=[c[0] for c in COLUMNS])
    writer.writeheader()
    writer.writerows(rows)

  header = [c[0] for c in COLUMNS]
  lines = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
  for row in rows:
    lines.append("| " + " | ".join(row[h] for h in header) + " |")
  md_path = os.path.join(REPO_ROOT, "results", "summary_table.md")
  with open(md_path, "w") as fh:
    fh.write("\n".join(lines) + "\n")

  print("\n".join(lines))
  print("\n-> %s\n-> %s" % (csv_path, md_path))


if __name__ == "__main__":
  main()
