import csv, io, json, os
import numpy as np

root = r"C:\Users\88950\Doubao\chats\2026-11-13\new-chat-1\robotic_imitation_improved"
d = os.path.join(root, "results", "train_1e7", "phase_diag")
ARMS = ("pre", "post")
DIR = {"pre": "1e7_phasediag_pre__phasegrid+randenv",
       "post": "1e7_phasediag_post__phasegrid+randenv"}


def load_summary(arm):
    return json.load(io.open(os.path.join(d, DIR[arm], "summary.json"),
                             encoding="utf-8"))


def load_csv(arm, ep):
    return list(csv.DictReader(io.open(
        os.path.join(d, DIR[arm], "ep%02d.csv" % ep), encoding="utf-8")))


def load_joints(arm, ep):
    return list(csv.DictReader(io.open(
        os.path.join(d, DIR[arm], "ep%02d_joints.csv" % ep), encoding="utf-8")))


def first_sustained(rows, key, threshold, n=2, absolute=True, direction=1):
    hit = 0
    for i, r in enumerate(rows):
        v = float(r[key])
        v = abs(v) if absolute else v
        ok = v >= threshold if direction > 0 else v <= threshold
        hit = hit + 1 if ok else 0
        if hit >= n:
            return int(r["step"]) - (n - 1)
    return None


def onset(rows, joints):
    base_z = float(np.median([float(r["sim_z_m"]) for r in rows[:5]]))
    jmax = {}
    for r in joints:
        jmax[int(r["step"])] = max(jmax.get(int(r["step"]), 0.0),
                                   float(r["pose_err_rad"]))
    out = {
        "t_pitch_0.30": first_sustained(rows, "pitch_rad", 0.30),
        "t_root_0.20m": first_sustained(rows, "root_dist_m", 0.20),
        "t_joint_0.50rad": first_sustained(
            [{"step": s, "v": v} for s, v in sorted(jmax.items())], "v", 0.50),
        "t_height_drop_0.10m": next(
            (int(r["step"]) for r in rows if float(r["sim_z_m"]) <= base_z - 0.10),
            None),
        "t_vz_-1.0": first_sustained(rows, "lin_vz_mps", -1.0,
                                     direction=-1),
        "t_nonfoot_contact": next(
            (int(r["step"]) for r in rows if int(r["nonfoot_contacts"]) > 0),
            None),
        "base_z_m": base_z,
    }
    candidates = [(v, k) for k, v in out.items()
                  if k.startswith("t_") and v is not None]
    if candidates:
        step, key = min(candidates)
        out["earliest_step"] = step
        out["earliest_signal"] = key
        top = sorted([(float(j["pose_err_rad"]), int(j["joint"]))
                      for j in joints if int(j["step"]) == step], reverse=True)[:3]
        out["top_joints_at_onset"] = [{"joint": j, "err": e} for e, j in top]
    else:
        out["earliest_step"] = None
        out["earliest_signal"] = None
        out["top_joints_at_onset"] = []
    return out


report = {"arms": {}, "paired": []}
for arm in ARMS:
    s = load_summary(arm)
    eps = []
    for i, e in enumerate(s["episodes"]):
        rows = load_csv(arm, i)
        joints = load_joints(arm, i)
        o = onset(rows, joints)
        eps.append({
            "ep": i, "seed": e["seed"], "phase": e["start_phase"],
            "steps": e["steps"],
            "survived": bool(e["steps"] >= 41 and e["termination"] != "early_termination"),
            "causes": e["termination_causes"], "onset": o,
            "randomizer_params": e.get("randomizer_params"),
            "env_dt": e["env_time_step_s"],
        })
    report["arms"][arm] = {"randomizer_count": s["protocol"]["randomizer_count"],
                           "episodes": eps}

# paired by (phase, seed)
for a in report["arms"]["pre"]["episodes"]:
    b = next(x for x in report["arms"]["post"]["episodes"]
             if abs(x["phase"] - a["phase"]) < 1e-9 and x["seed"] == a["seed"])
    report["paired"].append({
        "phase": a["phase"], "seed": a["seed"],
        "pre_steps": a["steps"], "post_steps": b["steps"],
        "pre_survived": a["survived"], "post_survived": b["survived"],
        "pre_causes": a["causes"], "post_causes": b["causes"],
        "pre_onset": a["onset"], "post_onset": b["onset"],
        "params_identical": a["randomizer_params"] == b["randomizer_params"],
    })

path = os.path.join(d, "phase_diag_analysis.json")
io.open(path, "w", encoding="utf-8", newline="\n").write(
    json.dumps(report, indent=2))

print("params identical:", all(p["params_identical"] for p in report["paired"]))
for arm in ARMS:
    print("\n==", arm)
    for e in report["arms"][arm]["episodes"]:
        print("seed=%d phase=%.3f steps=%2d succ=%s earliest=%s@%s top=%s"
              % (e["seed"], e["phase"], e["steps"], e["survived"],
                 e["onset"]["earliest_signal"], e["onset"]["earliest_step"],
                 e["onset"]["top_joints_at_onset"]))
print("\n== paired onset")
for p in report["paired"]:
    print("phase=%.3f seed=%d steps %d->%d | %s@%s -> %s@%s | %s -> %s"
          % (p["phase"], p["seed"], p["pre_steps"], p["post_steps"],
             p["pre_onset"]["earliest_signal"], p["pre_onset"]["earliest_step"],
             p["post_onset"]["earliest_signal"], p["post_onset"]["earliest_step"],
             ",".join(p["pre_causes"] or []), ",".join(p["post_causes"] or [])))
print(path)
