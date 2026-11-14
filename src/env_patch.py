"""Opt-in improvements layered on the vendored upstream environment.

The upstream sources under ``third_party/motion_imitation`` are never edited.
Every change below is a monkey-patch applied to a *live* environment object,
so the baseline behaviour (all flags off) follows the upstream code path, and
each improvement can be switched on independently. Alignment with upstream
``eval_pretrained.py`` is summary-level only (dog_pace 0.7496, dog_spin 0.6378):
no per-step (state, obs, action, reward) regression exists, and pybullet here is
3.2.0 while upstream pins 2.7.3, so "bit-identical" would be an overclaim.

Improvements implemented here
-----------------------------
1. ``enable_cycle_heading_anchor`` -- upstream re-anchors the reference motion
   origin every gait cycle but only for *position*; rotation is re-anchored
   only when a new clip is switched in (``imitation_task.py``:
   ``_sync_ref_origin(sync_root_position=True, sync_root_rotation=change_clip)``).
   For cyclic skills such as ``dog_spin`` a small per-cycle rotation lag therefore
   integrates without bound (~0.05 rad/s in our measurements). Re-anchoring the
   heading each cycle turns an unbounded drift into a bounded per-cycle error
   without any retraining.

2. ``apply_velocity_reward_v2`` -- the upstream velocity term is
   ``exp(-0.1 * sum_j (dq_ref - dq_sim)^2)`` over 12 joints. Because the error is
   summed rather than averaged and the scale is small, the term saturates near
   zero (mean 2.6e-3, median 1.6e-4, p90 7.5e-3 over 5400 dog_pace steps): its
   gradient is negligible next to the pose term and it stops shaping the policy.
   v2 averages the error per joint and widens the scale (0.25) so the same
   operating range maps to median 0.162 (p10 0.048, p90 0.361, mean 0.191), then
   re-balances the five weights so they still sum to 1 (which
   keeps the per-step reward in [0, 1] and therefore keeps "per-step reward ==
   normalised return").
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

REWARD_WEIGHTS_V1 = {
    "pose": 0.50,
    "velocity": 0.05,
    "end_effector": 0.20,
    "root_pose": 0.15,
    "root_velocity": 0.10,
}

# Pose/end-effector/root terms saturate too, but they carry real signal; the
# budget for the velocity term comes out of the pose term, which was the only
# term with slack.
REWARD_WEIGHTS_V2 = {
    "pose": 0.45,
    "velocity": 0.10,
    "end_effector": 0.20,
    "root_pose": 0.15,
    "root_velocity": 0.10,
}

VELOCITY_ERR_SCALE_V1 = 0.1
# Measured on dog_pace with the pretrained policy: sum_j (dq_ref - dq_sim)^2 has
# median 87.3 (p10 48.9, p90 146.0, mean 92.1), i.e. 2.7 rad/s per-joint RMS.
# Upstream exp(-0.1 * sum) therefore returns a mean of 2.6e-3 (median 1.6e-4,
# p90 7.5e-3): saturated over almost the whole operating range, so it stops
# shaping the policy. exp(-0.1 * 92.1) = 1.0e-4 is only what plugging the mean
# error in gives; Jensen makes the realised mean 26x higher. v2 applies the
# *mean* per-joint error with scale 0.25, mapping the same range to median 0.162
# (p10 0.048, p90 0.361, mean 0.191): same monotone objective,
# but with a usable gradient. Note that 12 * 0.1 == 1.2, so merely switching to
# the mean (the naive "fix") is an exact identity -- the rescale is the actual
# change.
VELOCITY_ERR_SCALE_V2 = 0.25


def imitation_task(env):
  """Return the ImitationTask living inside a (possibly wrapped) env."""
  task = getattr(env, "_task", None)
  if task is not None and hasattr(task, "_calc_reward_pose"):
    return task
  inner = getattr(env, "env", None)
  if inner is not None:
    return imitation_task(inner)
  raise AttributeError("no ImitationTask found on this environment")


def enable_cycle_heading_anchor(env):
  """Re-anchor reference heading at every gait-cycle boundary (improvement 1)."""
  task = imitation_task(env)
  if getattr(task, "_heading_anchor_installed", False):
    return task

  def _update_ref_motion():
    time = task._get_motion_time()
    change_clip = task._check_change_clip()

    if change_clip:
      new_motion_id = task._sample_ref_motion()
      task._change_ref_motion(new_motion_id)
      task._reset_clip_change_time()
      task._motion_time_offset = task._sample_time_offset()

    motion = task.get_active_motion()
    new_phase = motion.calc_phase(time)

    if ((task._enable_cycle_sync and (new_phase < task._prev_motion_phase))
        or change_clip):
      # upstream: sync_root_rotation=change_clip
      task._sync_ref_origin(sync_root_position=True, sync_root_rotation=True)

    task._update_ref_state()
    task._update_ref_model()
    task._prev_motion_phase = new_phase

  task._update_ref_motion = _update_ref_motion
  task._heading_anchor_installed = True
  return task


def apply_velocity_reward_v2(env, weights=None, err_scale=None):
  """Rebalance the velocity term and the five reward weights (improvement 2)."""
  task = imitation_task(env)
  weights = weights or REWARD_WEIGHTS_V2
  if abs(sum(weights.values()) - 1.0) > 1e-9:
    raise ValueError("reward weights must sum to 1, got %r" % (weights,))
  err_scale = VELOCITY_ERR_SCALE_V2 if err_scale is None else err_scale

  task._pose_weight = weights["pose"]
  task._velocity_weight = weights["velocity"]
  task._end_effector_weight = weights["end_effector"]
  task._root_pose_weight = weights["root_pose"]
  task._root_velocity_weight = weights["root_velocity"]
  task._velocity_err_scale = err_scale
  task._velocity_reward_v2 = True

  def _calc_reward_velocity():
    env_ = task._env
    pyb = task._get_pybullet_client()
    sim_model = env_.robot.quadruped
    ref_model = task._ref_model

    vel_err = 0.0
    num = 0
    for j in range(task._get_num_joints()):
      ref = np.array(pyb.getJointStateMultiDof(ref_model, j)[1])
      sim = np.array(pyb.getJointStateMultiDof(sim_model, j)[1])
      if len(ref) > 0:
        diff = ref - sim
        vel_err += diff.dot(diff)
        num += 1

    if num == 0:
      return 1.0
    return float(np.exp(-err_scale * vel_err / num))

  task._calc_reward_velocity = _calc_reward_velocity
  return task


def attach_term_logger(env):
  """Capture the five reward terms *inside* ``task.reward()``.

  Calling ``task._calc_reward_*()`` from the outside after ``env.step()``
  returns values for the *next* reference state (the task advances its motion
  after computing the reward), so an external recomputation never adds up to
  the logged reward. Wrapping ``reward`` tees the exact terms the environment
  used, which makes the per-step reward verifiable against the five terms.
  """
  task = imitation_task(env)
  if getattr(task, "_term_logger_installed", False):
    return task
  orig_reward = task.reward

  def reward(env_):
    task._last_terms = {
        "pose": float(task._calc_reward_pose()),
        "velocity": float(task._calc_reward_velocity()),
        "end_effector": float(task._calc_reward_end_effector()),
        "root_pose": float(task._calc_reward_root_pose()),
        "root_velocity": float(task._calc_reward_root_velocity()),
    }
    return orig_reward(env_)

  task.reward = reward
  task._term_logger_installed = True
  return task


def set_start_phase(env, task, phase):
  """Pin the next episode to a fixed reference phase.

  ``phase`` is in [0, 1); it overrides the time offset that
  ``ImitationTask._reset_motion_time_offset`` would otherwise sample. Phases are
  generated by the caller from an explicit RNG so that "random phase" runs are
  reproducible across processes (the upstream sampler draws from an RNG whose
  state we cannot pin from the outside).
  """
  task._forced_start_phase = float(phase)
  if getattr(task, "_forced_phase_installed", False):
    return task

  orig_reset = task._reset_motion_time_offset

  def _reset_motion_time_offset():
    motion = task.get_active_motion()
    task._motion_time_offset = task._forced_start_phase * motion.get_duration()

  task._reset_motion_time_offset_orig = orig_reset
  task._reset_motion_time_offset = _reset_motion_time_offset
  task._forced_phase_installed = True
  return task


def clear_start_phase(task):
  """Restore the environment's own time-offset sampling."""
  orig = getattr(task, "_reset_motion_time_offset_orig", None)
  if orig is not None:
    task._reset_motion_time_offset = orig
    task._forced_phase_installed = False
  return task
