"""Build the 20-pair pre/post comparison from two evaluate.py summaries.

Both runs must use the same phase list and the same (randomizer-off) protocol;
the script pairs records by start phase and reports paired transitions,
paired differences and exact McNemar.
"""

import argparse
import io
import json
import math
import os

import numpy as np

FIELDS = ("mean_reward", "mean_pose_err_rad", "mean_root_err_m",
          "mean_yaw_err_rad", "final_yaw_err_rad", "mean_speed_mps",
          "max_yaw_err_rad")


def load(path):
    return json.load(io.open(path, encoding="utf-8"))


def is_success(ep):
    return bool(ep["survived"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", required=True)
    ap.add_argument("--post", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pre = load(args.pre)
    post = load(args.post)
    by_phase_pre = {round(e["start_phase"], 9): e for e in pre["episodes"]}
    by_phase_post = {round(e["start_phase"], 9): e for e in post["episodes"]}
    phases = sorted(set(by_phase_pre) & set(by_phase_post))

    pairs = []
    for ph in phases:
        a, b = by_phase_pre[ph], by_phase_post[ph]
        row = {"phase": ph, "pre_steps": a["steps"], "post_steps": b["steps"],
               "pre_succ": is_success(a), "post_succ": is_success(b),
               "pre_causes": a["termination_causes"],
               "post_causes": b["termination_causes"],
               "pre_final_root_dist": a.get("final_root_dist_m"),
               "post_final_root_dist": b.get("final_root_dist_m"),
               "pre_final_rot": a.get("final_root_rot_angle_rad"),
               "post_final_rot": b.get("final_root_rot_angle_rad"),
               "pre": {k: a.get(k) for k in FIELDS},
               "post": {k: b.get(k) for k in FIELDS}}
        pairs.append(row)

    transitions = {
        "both_success": sum(p["pre_succ"] and p["post_succ"] for p in pairs),
        "pre_fail_post_success": sum((not p["pre_succ"]) and p["post_succ"]
                                      for p in pairs),
        "pre_success_post_fail": sum(p["pre_succ"] and (not p["post_succ"])
                                      for p in pairs),
        "both_fail": sum((not p["pre_succ"]) and (not p["post_succ"])
                         for p in pairs),
    }

    diffs = {}
    for key in ("steps",) + FIELDS:
        if key == "steps":
            values = [p["post_steps"] - p["pre_steps"] for p in pairs]
        else:
            values = [p["post"][key] - p["pre"][key] for p in pairs
                      if p["post"][key] is not None and p["pre"][key] is not None]
        arr = np.asarray(values, dtype=float)
        diffs[key] = {"mean": float(arr.mean()), "se": float(arr.std(ddof=1) / math.sqrt(len(arr))),
                      "min": float(arr.min()), "max": float(arr.max()), "n": int(len(arr))}

    b = transitions["pre_fail_post_success"]
    c = transitions["pre_success_post_fail"]
    n = b + c
    # exact two-sided McNemar: 2 * P(X <= min(b, c)) under Binomial(n, 0.5)
    k = min(b, c)
    p_exact = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

    result = {"n_pairs": len(pairs),
              "pre_success": sum(p["pre_succ"] for p in pairs),
              "post_success": sum(p["post_succ"] for p in pairs),
              "transitions": transitions,
              "mcnemar_exact_p": p_exact,
              "paired_diffs": diffs,
              "pairs": pairs}
    io.open(args.out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(result, indent=2))

    print("pairs=%d pre_success=%d post_success=%d mcnemar_p=%.3f"
          % (len(pairs), result["pre_success"], result["post_success"], p_exact))
    for k, v in diffs.items():
        print("%-20s mean=%+.4f se=%.4f t=%.2f"
              % (k, v["mean"], v["se"], abs(v["mean"]) / v["se"]))
    print(args.out)


if __name__ == "__main__":
    main()
