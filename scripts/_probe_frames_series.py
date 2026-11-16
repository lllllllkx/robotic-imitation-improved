"""Dump reference-yaw series with and without the cycle heading anchor.

The reference trajectory is a function of time only, so the anchored and the
baseline run visit the same clip frames at the same step index (same start
phase, same episode duration). That makes the counterfactual checkable
directly: baseline psi_ref(k) should equal anchored psi_ref(k) - cum(k) if the
reconstruction is right.

Usage: python scripts/_probe_frames_series.py <anchor|baseline> <out.npz>
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import env_patch  # noqa: E402
from src import metrics  # noqa: E402
from src.evaluate import DEFAULT_CODE_DIR  # noqa: E402
from src.evaluate import MOTION_DIR  # noqa: E402
from src.evaluate import POLICY_DIR  # noqa: E402
from src.evaluate import build_env  # noqa: E402

MODE = sys.argv[1]
OUT = sys.argv[2]
MOTION = "dog_spin"
STEPS = 240
PHASE = 0.25

motion_file = os.path.join(MOTION_DIR, MOTION + ".txt")
model_file = os.path.join(POLICY_DIR, MOTION + ".zip")
env, model, _ = build_env(DEFAULT_CODE_DIR, motion_file, model_file)
task = env_patch.imitation_task(env)
task._ref_state_init_prob = 1.0
task._enable_rand_init_time = False
if MODE == "anchor":
  env_patch.enable_cycle_heading_anchor(env)
env_patch.attach_term_logger(env)

env.seed(0)
env_patch.pin_task_random(task, 0)
env_patch.set_start_phase(env, task, PHASE)
o = env.reset()

rec = {"ref_time": [], "psi_ref": [], "psi_sim": [], "cum": [], "theta": [],
       "reward": []}
for k in range(STEPS):
  a, _ = model.predict(o, deterministic=True)
  o, r, done, info = env.step(a)
  snap = task._last_snapshot
  rec["ref_time"].append(snap["ref_time"])
  rec["psi_ref"].append(metrics.quat_to_yaw(snap["ref_orn"]))
  rec["psi_sim"].append(metrics.quat_to_yaw(snap["sim_orn"]))
  rec["cum"].append(float(snap.get("anchor_cum_yaw", 0.0)))
  rec["theta"].append(float(snap.get("anchor_theta_yaw", 0.0)))
  rec["reward"].append(float(r))
  if done:
    print("done at", k)
    break

np.savez(OUT, **{k: np.asarray(v) for k, v in rec.items()})
print("%s: %d steps, psi_ref[0]=%.4f psi_ref[-1]=%.4f cum[-1]=%.4f "
      "mean_reward=%.4f" % (MODE, len(rec["reward"]), rec["psi_ref"][0],
                            rec["psi_ref"][-1], rec["cum"][-1],
                            np.mean(rec["reward"])))
env.close()
