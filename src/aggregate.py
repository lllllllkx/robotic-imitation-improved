"""Collect evaluation summaries into one comparison table.

Scans the *whole* results tree (``results/<group>/<skill>__<variant>/summary.json``)
rather than only ``results/eval``, so the deterministic controls written to
``results/det_*`` are not silently left out of the table that the one-shot
protocol script produces. Each row records which ``group`` directory it came
from (``main`` for ``results/eval``, ``det`` for ``results/det_*``).

Outputs
-------
results/summary_table.csv   one row per (group, skill, variant, init)
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

GROUP_ALIASES = {"eval": "main"}


def group_of(path, results_root):
  """Name of the first directory level below ``results_root``."""
  rel = os.path.relpath(path, results_root)
  parts = rel.split(os.sep)
  name = parts[0] if len(parts) > 1 else "main"
  if name.startswith("det"):
    return "det"
  return GROUP_ALIASES.get(name, name)


def fmt_opt(value, digits=4):
  """Format an optional aggregate entry; missing instrumentation -> n/a."""
  if value is None:
    return "n/a"
  return ("%% .%df" % digits % value).strip()


COLUMNS = [
    ("group", lambda p: p["_group"]),
    ("skill", lambda p: p["tag"]),
    ("variant", lambda p: p["variant"]),
    ("init", lambda p: p.get("protocol", {}).get("init_mode", "controlled")),
    ("episodes", lambda p: str(p["aggregate"]["num_episodes"])),
    ("mean_reward", lambda p: "%.4f" % p["aggregate"]["mean_reward_mean"]),
    ("std_reward", lambda p: "%.4f" % p["aggregate"]["mean_reward_std"]),
    ("seed_mean", lambda p: "%.4f" % p["by_seed"]["mean_reward_mean_across_seeds"]),
    ("seed_std", lambda p: "%.4f" % p["by_seed"]["mean_reward_std_across_seeds"]),
    ("pose_err_rad", lambda p: "%.4f" % p["aggregate"]["mean_pose_err_rad_mean"]),
    ("root_err_m", lambda p: "%.4f" % p["aggregate"]["mean_root_err_m_mean"]),
    ("yaw_err_rad", lambda p: "%.4f" % p["aggregate"]["mean_yaw_err_rad_mean"]),
    ("yaw_drift_rad_s",
     lambda p: "%.4f" % p["aggregate"]["yaw_drift_rad_per_s_mean"]),
    # Fixed-frame (post-hoc) readings. For runs without --yaw-anchor the
    # accumulated offset is 0 by construction, so only the anchored variants
    # carry a different number here; "n/a" marks a summary that predates the
    # frame instrumentation and must not be read as zero.
    ("yaw_fixed_rad", lambda p: fmt_opt(
        p["aggregate"].get("mean_yaw_err_fixed_rad_mean"))),
    ("anchor_cum_rad", lambda p: fmt_opt(
        p["aggregate"].get("final_anchor_cum_yaw_rad_mean"))),
    ("cum_gap_rad", lambda p: fmt_opt(
        p["aggregate"].get("max_abs_cum_tracker_vs_geometry_rad_mean"))),
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


def find_summaries(results_root, dirs=None):
  """All ``summary.json`` files under the given directories (or the whole tree)."""
  if dirs:
    out = []
    for d in dirs:
      out.extend(sorted(glob.glob(os.path.join(d, "*", "summary.json"))))
    return out
  return sorted(glob.glob(os.path.join(results_root, "*", "*", "summary.json")))


def collect(results_root, run_id=None, dirs=None, only_groups=None,
            verbose=True):
  rows = []
  skipped = []
  all_ids = set()
  for path in find_summaries(results_root, dirs):
    payload = load(path)
    group = group_of(path, results_root)
    if group.startswith("_"):
      # scratch directories (results/_smoke*, results/_probe*) are gitignored
      # probes, never part of a reported table
      continue
    if only_groups and group not in only_groups:
      continue
    all_ids.add(payload.get("run_id"))
    if run_id is not None and payload.get("run_id") != run_id:
      skipped.append(path)
      continue
    payload["_group"] = group
    rows.append({name: fn(payload) for name, fn in COLUMNS})
  order = {"main": 0, "det": 1}
  rows.sort(key=lambda r: (order.get(r["group"], 2), r["skill"],
                           r["group"] != "main", r["init"], r["variant"]))
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
  p.add_argument("--results-root", default=os.path.join(REPO_ROOT, "results"),
                 help="root scanned for <group>/<run>/summary.json")
  p.add_argument("--dir", action="append", default=None,
                 help="explicit results directory (repeatable); overrides the "
                      "recursive scan")
  p.add_argument("--group", action="append", default=None,
                 help="only include these groups (main, det)")
  p.add_argument("--run-id", default=None,
                 help="only include summaries produced with this run id")
  p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "results"))
  # kept for backwards compatibility with earlier docs/commands
  p.add_argument("--results-dir", default=None, help=argparse.SUPPRESS)
  args = p.parse_args(argv)

  dirs = args.dir
  if dirs is None and args.results_dir is not None:
    dirs = [args.results_dir]

  rows = collect(args.results_root, args.run_id, dirs, args.group)
  if not rows:
    raise SystemExit("no summaries under %s (run-id filter: %s)"
                     % (args.results_root, args.run_id))

  os.makedirs(args.out_dir, exist_ok=True)
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
