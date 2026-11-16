"""List every reference-origin sync event with its tag and resulting offset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import env_patch  # noqa: E402
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

marker = [0]


def report():
  ev = task._anchor_tracker.events
  while marker[0] < len(ev):
    shared, theta, cum = ev[marker[0]]
    print("event %2d step=%3d shared=%-5s theta=%+.4f cum=%+.4f"
          % (marker[0], step_now[0], shared, theta, cum))
    marker[0] += 1


step_now = [0]
o = env.reset()
report()
for k in range(80):
  a, _ = model.predict(o, deterministic=True)
  o, r, done, info = env.step(a)
  step_now[0] = k
  report()
  if done:
    print("done at", k)
    break
print("events total", len(task._anchor_tracker.events))
env.close()
