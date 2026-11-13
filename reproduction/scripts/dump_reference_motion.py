"""Dump the reference motion clips shipped with the paper's repo to CSV.

These .txt files are the mocap-derived reference motions used in the paper;
this script makes them analysable without running the simulator.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np


def dump(code_dir, motion_file, out_csv):
  sys.path.insert(0, os.path.abspath(code_dir))
  if os.getcwd() != os.path.abspath(code_dir):
    os.chdir(os.path.abspath(code_dir))
  from utilities import motion_data

  motion = motion_data.MotionData(motion_file)
  frames = motion.get_frames()
  n = motion.get_num_frames()
  dt = motion.get_frame_duration()

  joint_cols = []
  rows = []
  for f in range(n):
    frame = frames[f]
    t = motion.get_frame_time(f)
    root_pos = motion.get_frame_root_pos(frame)
    root_rot = motion.get_frame_root_rot(frame)
    joints = motion.get_frame_joints(frame)
    if not joint_cols:
      joint_cols = ["q%d" % i for i in range(len(joints))]
    rows.append([t, root_pos[0], root_pos[1], root_pos[2],
                 root_rot[0], root_rot[1], root_rot[2], root_rot[3]]
                + list(joints))

  with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["time", "root_x", "root_y", "root_z",
                "root_qx", "root_qy", "root_qz", "root_qw"] + joint_cols)
    w.writerows(rows)

  speed = float(np.mean(np.linalg.norm(np.diff(frames[:, :3], axis=0), axis=1)
                        / dt))
  return {
      "motion_file": os.path.basename(motion_file),
      "num_frames": int(n),
      "frame_duration_s": float(dt),
      "duration_s": float(motion.get_duration()),
      "cycles": float(motion.get_cycle_count()) if hasattr(
          motion, "get_cycle_count") else None,
      "mean_speed_mps": speed,
      "root_height_mean": float(np.mean(frames[:, 2])),
      "root_height_min": float(np.min(frames[:, 2])),
      "root_height_max": float(np.max(frames[:, 2])),
      "num_joints": int(frames.shape[1] - 7),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--code_dir", required=True)
  ap.add_argument("--motion_dir", required=True)
  ap.add_argument("--out_dir", required=True)
  args = ap.parse_args()

  os.makedirs(args.out_dir, exist_ok=True)
  meta = []
  for path in sorted(glob.glob(os.path.join(args.motion_dir, "*.txt"))):
    name = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(args.out_dir, name + ".csv")
    info = dump(args.code_dir, path, csv_path)
    meta.append(info)
    print("[ref] %-22s frames=%4d dur=%.2fs speed=%.3f m/s" %
          (name, info["num_frames"], info["duration_s"],
           info["mean_speed_mps"]))
  with open(os.path.join(args.out_dir, "reference_summary.json"), "w") as f:
    json.dump(meta, f, indent=2)


if __name__ == "__main__":
  main()
