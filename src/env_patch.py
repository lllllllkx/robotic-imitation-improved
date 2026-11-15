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
   summed rather than averaged and the scale is small, the term sits near zero
   over almost the whole operating range (mean 2.6e-3, median 1.6e-4, p90 7.5e-3
   over 5400 dog_pace steps), so its weighted contribution to the per-step reward
   is 0.02%. v2 averages the error per joint and widens the scale (0.25), moving
   the same operating range to median 0.162 (p10 0.048, p90 0.361, mean 0.191)
   and lifting the weighted contribution to 2.6%; the five weights are
   re-balanced so they still sum to 1 (which keeps the per-step reward in
   [0, 1] and therefore keeps "per-step reward == normalised return").
   This is a statement about the *numerical range of the reward signal*, not
   about training: no end-to-end PPO retraining with v2 was run, so a claim
   about policy gradients or final performance would be unsupported.
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
# p90 7.5e-3): numerically flat over almost the whole operating range, so its
# weighted contribution is 0.02% of the per-step reward. exp(-0.1 * 92.1) =
# 1.0e-4 is only what plugging the mean error in gives; Jensen makes the
# realised mean 26x higher. v2 applies the *mean* per-joint error with scale
# 0.25, mapping the same range to median 0.162 (p10 0.048, p90 0.361,
# mean 0.191): the same monotone objective, but with visible numerical variation
# inside the operating range, and a weighted contribution of 2.6%. This
# quantifies the reward signal, not a training effect. Note that 12 * 0.1 ==
# 1.2, so merely switching to the mean (the naive "fix") is an exact identity --
# the rescale is the actual change.
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
  """Capture the five reward terms *and* a tracking snapshot inside ``reward()``.

  Upstream ``LocomotionGymEnv.step()`` runs ``robot.Step`` -> ``self._reward()``
  -> ``task.update()`` -> ``_update_ref_state()/_update_ref_model()``, i.e. the
  reference motion advances *after* the reward of the current step has been
  computed. Reading reference poses from outside ``env.step()`` therefore
  compares the robot's state against frame ``t+1`` while the reward it earned
  was measured against frame ``t`` -- a one-simulation-step offset. That offset
  also interacts with ``--yaw-anchor``: the heading is re-anchored inside
  ``_update_ref_motion``, so a post-step read would see an already-corrected
  reference.

  Wrapping ``reward`` tees both the exact terms and the exact (robot state,
  reference state) pair they were computed against. ``evaluate.py`` derives
  every tracking metric from this snapshot, so reward, joint error, root error,
  yaw error and drift all live on the same time point.
  """
  task = imitation_task(env)
  if getattr(task, "_term_logger_installed", False):
    return task
  orig_reward = task.reward

  pyb = task._get_pybullet_client()
  sim_model = env.robot.quadruped
  ref_model = task._ref_model
  n_joints = task._get_num_joints()

  def reward(env_):
    task._last_terms = {
        "pose": float(task._calc_reward_pose()),
        "velocity": float(task._calc_reward_velocity()),
        "end_effector": float(task._calc_reward_end_effector()),
        "root_pose": float(task._calc_reward_root_pose()),
        "root_velocity": float(task._calc_reward_root_velocity()),
    }
    # Reference frame this reward is scored against (before task.update()).
    # ``task._ref_pose`` is the frame tuple set by the last _update_ref_motion,
    # i.e. exactly the one the terms above were computed from.
    motion = task.get_active_motion()
    ref_joints = np.array(motion.get_frame_joints(task._ref_pose))
    sim_joints = []
    for j in range(n_joints):
      jq = pyb.getJointStateMultiDof(sim_model, j)[0]
      if len(jq) > 0:
        sim_joints.append(float(jq[0]))
    sim_pos, sim_orn = pyb.getBasePositionAndOrientation(sim_model)
    ref_pos, ref_orn = pyb.getBasePositionAndOrientation(ref_model)
    task._last_snapshot = {
        "ref_time": float(task._get_motion_time()),
        "ref_joints": ref_joints,
        "sim_joints": np.array(sim_joints),
        "sim_pos": np.array(sim_pos),
        "sim_orn": np.array(sim_orn),
        "ref_pos": np.array(ref_pos),
        "ref_orn": np.array(ref_orn),
    }
    return orig_reward(env_)

  task.reward = reward
  task._term_logger_installed = True
  return task


def pin_task_random(task, seed):
  """Give the task a private, seeded numpy RNG.

  Upstream samples the initial state (``ref_state_init_prob``) and the random
  start time through ``self._env.np_random`` of the *task*. ``env.seed()``
  reaches the gym env, but not the object the task holds, so with
  ``--upstream-init`` those draws fell through to the process-global
  ``np.random``: the same seed produced different rollouts in different
  processes (measured: three rollouts of seed 0 gave 0.7034 / 0.7050 / 0.6959,
  and one draw produced a reset that terminated on step 1). Pinning a
  RandomState per seed makes that regime reproducible without changing which
  distribution is sampled.
  """
  rng = np.random.RandomState(int(seed))
  task._env.np_random = rng
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
