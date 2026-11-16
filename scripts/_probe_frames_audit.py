"""Per-step audit of the anchored vs fixed reference heading."""

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

motion_file = os.path.join(MOTION_DIR, "dog_spin.txt")
model_file = os.path.join(POLICY_DIR, "dog_spin.zip")
env, model, _ = build_env(DEFAULT_CODE_DIR, motion_file, model_file)
task = env_patch.imitation_task(env)
task._ref_state_init_prob = 1.0
task._enable_rand_init_time = False
env_patch.enable_cycle_heading_anchor(env)
env_patch.attach_term_logger(env)
env.seed(0)
env_patch.pin_task_random(task, 0)
env_patch.set_start_phase(env, task, 0.25)
o = env.reset()
tr = env_patch.reset_anchor_tracker(env)
print("after reset: theta=%s theta_cf=%s" % (tr.theta, tr.theta_cf))
print("origin offset rot", np.round(task._origin_offset_rot, 5))

for k in range(40):
  a, _ = model.predict(o, deterministic=True)
  o, r, done, info = env.step(a)
  s = task._last_snapshot
  yaw_anch = metrics.quat_to_yaw(s["ref_orn"])
  yaw_raw = s["ref_yaw_raw"]
  theta = s["anchor_theta_yaw"]
  theta_cf = s["anchor_theta_cf_yaw"]
  fixed_orn = metrics.with_yaw_offset(s["ref_orn_raw"], theta_cf)
  yaw_fix = metrics.quat_to_yaw(fixed_orn)
  print("k=%2d theta=%+.4f cf=%+.4f cum=%+.4f yaw_anch=%+.4f yaw_raw=%+.4f "
        "yaw_fix=%+.4f  anch-fix=%+.4f  tracker_theta=%s"
        % (k, theta, theta_cf, s["anchor_cum_yaw"], yaw_anch, yaw_raw, yaw_fix,
           metrics.wrap_angle(yaw_anch - yaw_fix),
           round(tr.theta, 4) if tr.theta is not None else None))
  if done:
    break
env.close()
