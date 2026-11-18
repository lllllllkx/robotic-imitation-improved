"""Onset analysis for the canonical (randomizer-off) failure trajectories.

Reads one or two evaluate.py output directories that contain epNN.csv and
epNN_joints.csv, and reports the first step at which each physically
meaningful precursor appears.
"""

import argparse
import csv
import io
import json
import os

import numpy as np


def first_sustained(rows, key, threshold, n=2, direction=1):
    hit = 0
    for r in rows:
        v = float(r[key])
        ok = v >= threshold if direction > 0 else v <= threshold
        hit = hit + 1 if ok else 0
        if hit >= n:
            return int(r["step"]) - (n - 1)
    return None


def episode_onset(csv_path, joints_path):
    rows = list(csv.DictReader(io.open(csv_path, encoding="utf-8")))
    joints = list(csv.DictReader(io.open(joints_path, encoding="utf-8")))
    base_z = float(np.median([float(r["sim_z_m"]) for r in rows[:8]]))
    jmax = {}
    for j in joints:
        st = int(j["step"])
        jmax[st] = max(jmax.get(st, 0.0), float(j["pose_err_rad"]))
    jrows = [{"step": st, "v": v} for st, v in sorted(jmax.items())]

    onset = {
        "t_root_0.20m": first_sustained(rows, "root_dist_m", 0.20),
        "t_joint_0.50rad": first_sustained(jrows, "v", 0.50),
        "t_height_drop_0.10m": next(
            (int(r["step"]) for r in rows
             if float(r["sim_z_m"]) <= base_z - 0.10), None),
        "t_nonfoot_contact": next(
            (int(r["step"]) for r in rows if int(r["nonfoot_contacts"]) > 0),
            None),
        "base_z_m": base_z,
    }
    step = None
    for k in ("t_root_0.20m", "t_joint_0.50rad",
              "t_height_drop_0.10m", "t_nonfoot_contact"):
        if onset[k] is not None:
            step = onset[k] if step is None else min(step, onset[k])
    onset["earliest_step"] = step
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory containing arm subdirs")
    ap.add_argument("--pre", required=True, help="pre arm subdirectory name")
    ap.add_argument("--post", required=True, help="post arm subdirectory name")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = {}
    for arm, sub in (("pre", args.pre), ("post", args.post)):
        base = os.path.join(args.dir, sub)
        summary = json.load(io.open(os.path.join(base, "summary.json"),
                                    encoding="utf-8"))
        rows = []
        for i, e in enumerate(summary["episodes"]):
            onset = episode_onset(
                os.path.join(base, "ep%02d.csv" % i),
                os.path.join(base, "ep%02d_joints.csv" % i))
            terminal = [
                {"joint": int(j["joint"]), "err": float(j["pose_err_rad"])}
                for j in csv.DictReader(io.open(
                    os.path.join(base, "ep%02d_joints.csv" % i),
                    encoding="utf-8"))
                if int(j["step"]) == e["steps"]]
            terminal.sort(key=lambda x: -x["err"])
            rows.append({"ep": i, "phase": e["start_phase"], "steps": e["steps"],
                         "survived": e["survived"],
                         "causes": e["termination_causes"],
                         "onset": onset, "joint_top_terminal": terminal[:3]})
        result[arm] = rows

    io.open(args.out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(result, indent=2))
    for arm in ("pre", "post"):
        print("==", arm)
        for e in result[arm]:
            print("phase=%.4f steps=%2d survived=%s %s"
                  % (e["phase"], e["steps"], e["survived"], e["onset"]))
    print(args.out)


if __name__ == "__main__":
    main()
