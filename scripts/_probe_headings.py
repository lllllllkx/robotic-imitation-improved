"""Probe: is psi_ref == psi_raw + theta, and is cum == theta - theta_cf?

Runs the anchored environment for a few dozen steps and prints, at every step:
  psi_raw  heading of the clip frame with the origin offset NOT applied
  theta    world yaw of _origin_offset_rot
  psi_ref  heading of the reference model at reward time
  psi_sim  heading of the robot at reward time
  cum      tracker's accumulated offset
The identity psi_ref == wrap(psi_raw + theta) is the whole basis of the
fixed-frame reconstruction, so it is checked numerically here rather than
assumed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import env_patch
from src import metrics
from src.evaluate import DEFAULT_CODE_DIR
from src.evaluate import MOTION_DIR
from src.evaluate import POLICY_DIR
from src.evaluate import build_env

MOTION = "dog_spin"
STEPS = 120

motion_file = os.path.join(MOTION_DIR, MOTION + ".txt")
model_file = os.path.join(POLICY_DIR, MOTION + ".zip")
env, model, _ = build_env(DEFAULT_CODE_DIR, motion_file, model_file)
task = env_patch.imitation_task(env)
task._ref_state_init_prob = 1.0
task._enable_rand_init_time = False
env_patch.enable_cycle_heading_anchor(env)
env_patch.attach_term_logger(env)

import numpy as np  # noqa: E402

env.seed(0)
env_patch.pin_task_random(task, 0)
env_patch.set_start_phase(env, task, 0.25)
o = env.reset()

motion = task.get_active_motion()
rows = []
for k in range(STEPS):
  a, _ = model.predict(o, deterministic=True)
  o, r, done, info = env.step(a)
  snap = task._last_snapshot
  psi_ref = metrics.quat_to_yaw(snap["ref_orn"])
  psi_sim = metrics.quat_to_yaw(snap["sim_orn"])
  theta = snap["anchor_theta_yaw"]
  cum = snap["anchor_cum_yaw"]
  # raw clip heading with the origin offset not applied
  try:
    raw_pose = task._calc_ref_pose(task._get_motion_time(), apply_origin_offset=False)
    psi_raw = metrics.quat_to_yaw(
        np.array(motion.get_frame_root_rot(raw_pose)))
  except Exception as exc:  # noqa: BLE001
    psi_raw = float("nan")
    print("raw pose failed:", exc)
  residual = metrics.wrap_angle(psi_ref - metrics.wrap_angle(psi_raw + theta))
  rows.append((k, psi_raw, theta, psi_ref, psi_sim, cum, residual))
  if k % 10 == 0 or k < 3:
    print("k=%3d raw=%+.4f theta=%+.4f ref=%+.4f sim=%+.4f cum=%+.4f "
          "resid=%+.2e" % (k, psi_raw, theta, psi_ref, psi_sim, cum, residual))
  if done:
    print("done at", k)
    break

print("tracker: cycle_events=%d shared=%d clip_delta=%.4f theta=%.4f"
      % (task._anchor_tracker.cycle_events, task._anchor_tracker.shared_events,
         task._anchor_tracker.clip_sync_delta, task._anchor_tracker.theta))
maxres = max(abs(r[6]) for r in rows)
print("max |psi_ref - (psi_raw+theta)| = %.3e" % maxres)
print("psi_sim advance = %.4f, psi_raw advance = %.4f, theta change = %.4f"
      % (metrics.wrap_angle(rows[-1][4] - rows[0][4]),
         metrics.wrap_angle(rows[-1][1] - rows[0][1]),
         metrics.wrap_angle(rows[-1][2] - rows[0][2])))
print("first-frame psi_raw=%.4f theta_cf=%.4f" % (rows[0][1], rows[0][2]))
env.close()
