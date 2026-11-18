"""Extended evaluation for motion-imitation policies.

What this adds on top of the upstream ``eval_pretrained.py`` loop (see
``docs/IMPROVEMENTS.md``):

* multi-episode runs with mean/std instead of a single rollout, with an
  optional random start phase along the reference clip;
* two explicit initial-state regimes: ``--upstream-init`` keeps the upstream
  defaults (90% chance to initialise from the reference state, randomised
  start time); without it the environment runs in *controlled* mode (always
  initialise from the reference state, phase set by us) so comparisons are
  paired;
* the five reward terms are logged inside ``task.reward()`` and
  ``mean(reward)`` is cross-checked against the normalised return rebuilt
  from those terms;
* yaw from the full quaternion, reporting both a bounded tracking error
  (wrapped absolute) and a drift slope fitted on the *unwrapped signed*
  error;
* with ``--yaw-anchor``, a second yaw series in the *fixed reference frame*:
  the cycle re-anchoring rotates the reference heading, so the same robot
  pose is also scored against the frame the reference would have had without
  the extra cycle syncs (``ref_yaw - anchor_cum_yaw``). This is a post-hoc
  reconstruction of one executed trajectory, not the error a policy would
  produce with anchoring disabled (``docs/HEADING_FRAMES.md``);
* horizontal and 3-D speed, an explicit termination reason, and provenance
  metadata (policy SHA256, library versions, upstream commit).

This protocol is *not* the paper's. The paper's Table V averages normalised
return over **three independently trained policies** evaluated in simulation
with canonical dynamics; here a single released policy is evaluated over
several initial conditions. Seed indices below therefore select initial
conditions, not training runs.

Usage (from the repository root, inside the project's conda env):

  python -m src.evaluate --motion dog_pace --episodes 5 --seeds 3 --random-phase --yaw-anchor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
import time

# Pin BLAS/OMP pools *before* numpy (or anything that pulls it in) is imported:
# once a thread pool is initialised these variables no longer apply, and a
# multi-threaded pool makes the same policy+seed diverge across processes at
# the 1e-7 level, which is enough to change a marginally stable 600-step
# rollout. See docs/IMPROVEMENTS.md ("reproducibility").
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
  os.environ[_var] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402  (must follow the thread pinning above)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CODE_DIR = os.path.join(REPO_ROOT, "third_party", "motion_imitation")
MOTION_DIR = os.path.join(DEFAULT_CODE_DIR, "data", "motions")
POLICY_DIR = os.path.join(DEFAULT_CODE_DIR, "data", "policies")
UPSTREAM_COMMIT = "2f206ab"  # commit vendored into third_party/motion_imitation

from src import env_patch  # noqa: E402  (repo root must be on sys.path)
from src import metrics  # noqa: E402


def parse_args(argv=None):
  p = argparse.ArgumentParser()
  p.add_argument("--code_dir", default=DEFAULT_CODE_DIR)
  p.add_argument("--motion", default=None,
                 help="reference motion name, e.g. dog_pace (uses data/motions)")
  p.add_argument("--motion_file", default=None)
  p.add_argument("--extra-motions", default=None,
                 help="comma separated extra motions from data/motions, e.g."
                      " dog_spin; more than one loaded clip enables multi-clip"
                      " episodes (needs --clip-change)")
  p.add_argument("--clip-change", default=None,
                 help="multi-clip: 'min:max' seconds between clip switches"
                      " (uniform draw per episode from the seeded task RNG)")
  p.add_argument("--policy", default=None,
                 help="pretrained policy name, e.g. dog_pace (uses data/policies)")
  p.add_argument("--model_file", default=None)
  p.add_argument("--episodes", type=int, default=5)
  p.add_argument("--seeds", type=int, default=1,
                 help="number of evaluation seeds (initial conditions, not "
                      "training seeds)")
  p.add_argument("--random-phase", action="store_true",
                 help="start each episode from a random point in the clip")
  p.add_argument("--start-phases", default=None,
                 help="comma separated phases in [0,1), overrides random phase")
  p.add_argument("--upstream-init", action="store_true",
                 help="keep the upstream initial-state distribution "
                      "(ref_state_init_prob=0.9, randomised start time) "
                      "instead of the controlled one")
  p.add_argument("--yaw-anchor", action="store_true",
                 help="improvement 1: re-anchor reference heading every cycle")
  p.add_argument("--reward-v2", action="store_true",
                 help="improvement 2: rebalanced velocity term and weights")
  p.add_argument("--max-steps", type=int, default=600)
  p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "eval"))
  p.add_argument("--tag", default=None)
  p.add_argument("--run-id", default=None,
                 help="tag this run so src.aggregate can select it later")
  p.add_argument("--csv", action="store_true", help="also dump per-step csv")
  p.add_argument("--enable-randomizer", action="store_true",
                 help="enable the training-time environment randomizer; each"
                      " episode is randomised with the episode seed")
  p.add_argument("--diag", action="store_true",
                 help="with --csv, also dump joint-level trajectories, base"
                      " rpy/height/velocity, foot vs non-foot contacts and the"
                      " termination-condition flags per step")
  p.add_argument("--quiet", action="store_true")
  return p.parse_args(argv)


def resolve_paths(args):
  motion_file = args.motion_file
  if motion_file is None:
    if args.motion is None:
      raise SystemExit("give --motion or --motion_file")
    motion_file = os.path.join(MOTION_DIR, args.motion + ".txt")
  motion_files = [motion_file]
  if args.extra_motions:
    for name in [x.strip() for x in args.extra_motions.split(",") if x.strip()]:
      extra = name if name.endswith(".txt") else os.path.join(
          MOTION_DIR, name + ".txt")
      motion_files.append(extra)
  if args.clip_change and len(motion_files) < 2:
    raise SystemExit("--clip-change needs more than one motion "
                     "(--extra-motions); upstream only switches clips when"
                     " num_motions > 1")
  model_file = args.model_file
  policy = args.policy
  if model_file is None:
    if policy is None:
      policy = os.path.splitext(os.path.basename(motion_file))[0]
    model_file = os.path.join(POLICY_DIR, policy + ".zip")
  for path in motion_files + [model_file]:
    if not os.path.isfile(path):
      raise SystemExit("missing file: %s" % path)
  return motion_files, model_file


def parse_clip_change(spec):
  """'min:max' seconds -> (min, max); None keeps the upstream no-change mode."""
  if not spec:
    return None
  try:
    lo, hi = [float(x) for x in spec.split(":")]
  except ValueError:
    raise SystemExit("--clip-change must look like '3:6' (seconds)")
  if not (0.0 < lo <= hi):
    raise SystemExit("--clip-change needs 0 < min <= max")
  return lo, hi


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def find_env_randomizers(env):
  """Return the LocomotionGymEnv randomizer list through the wrapper chain."""
  seen = set()
  stack = [env]
  while stack:
    obj = stack.pop()
    if id(obj) in seen:
      continue
    seen.add(id(obj))
    randomizers = getattr(obj, "_env_randomizers", None)
    if randomizers is not None:
      return randomizers
    for attr in ("_gym_env", "_env", "env", "_wrapped_env"):
      child = getattr(obj, attr, None)
      if child is not None and not isinstance(child, (str, bytes)):
        stack.append(child)
  return []


def jsonable(value):
  """Convert numpy scalars/arrays from randomizer parameter dumps to JSON."""
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, (np.floating, np.integer)):
    return value.item()
  if isinstance(value, (list, tuple)):
    return [jsonable(v) for v in value]
  return value


def termination_state(pyb, env, task, snap):
  """Which of the four upstream terminal conditions hold at this state."""
  foot_links = set(int(x) for x in env.robot.GetFootLinkIDs())
  ground = env.get_ground()
  contacts = (pyb.getContactPoints(bodyA=env.robot.quadruped, bodyB=ground)
              if env.env_step_counter > 0 else [])
  foot_contacts = sum(1 for c in contacts if int(c[3]) in foot_links)
  nonfoot_contacts = len(contacts) - foot_contacts
  root_dist = float(np.linalg.norm(np.asarray(snap["ref_pos"])
                                   - np.asarray(snap["sim_pos"])))
  rot_angle = metrics.quat_angle_between(snap["sim_orn"], snap["ref_orn"])
  # The robot model's rest orientation is not identity in world coordinates
  # (the base frame is rotated about x), so absolute roll/pitch would report
  # ~90 deg for an upright dog. Tilt has to be measured against the reference.
  q_rel = metrics.quat_multiply(metrics.quat_conjugate(snap["ref_orn"]),
                                snap["sim_orn"])
  q_rel = metrics.normalize_quat(q_rel)
  roll, pitch = metrics.quat_to_roll_pitch(q_rel)
  return {
      "motion_over": bool(task.is_motion_over()),
      "contact_fall": bool(nonfoot_contacts > 0),
      "root_pos_fail": bool(root_dist > 1.0),
      "root_rot_fail": bool(rot_angle > 0.5 * np.pi),
      "foot_contacts": foot_contacts,
      "nonfoot_contacts": nonfoot_contacts,
      "root_dist_m": root_dist,
      "root_rot_angle_rad": rot_angle,
      "roll_rad": roll,
      "pitch_rad": pitch,
  }


def build_env(code_dir, motion_files, model_file, enable_randomizer=False):
  sys.path.insert(0, os.path.abspath(code_dir))
  import tensorflow as tf  # noqa: F401
  import envs.env_builder as env_builder
  import learning.imitation_policies as imitation_policies
  # note: ppo_imitation pulls in mpi4py (MPI) even for single-process evaluation
  import learning.ppo_imitation as ppo_imitation

  if isinstance(motion_files, str):
    motion_files = [motion_files]
  env = env_builder.build_imitation_env(
      motion_files=list(motion_files),
      num_parallel_envs=1,
      mode="test",
      enable_randomizer=bool(enable_randomizer),
      enable_rendering=False)

  policy_kwargs = {"net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
                   "act_fun": tf.nn.relu}
  model = ppo_imitation.PPOImitation(
      policy=imitation_policies.ImitationPolicy,
      env=env, gamma=0.95, timesteps_per_actorbatch=4096, clip_param=0.2,
      optim_epochs=1, optim_stepsize=1e-5, optim_batchsize=256, lam=0.95,
      adam_epsilon=1e-5, schedule="constant", policy_kwargs=policy_kwargs,
      verbose=0)
  model.load_parameters(model_file)
  versions = {"python": sys.version.split()[0]}
  # pybullet 不导出 __version__，只能用发行包元数据；stable-baselines 同理。
  for name in ("tensorflow", "numpy", "pybullet", "gym", "stable_baselines", "mpi4py"):
    try:
      module = __import__(name)
    except Exception:  # noqa: BLE001  (metadata only, never fatal)
      versions[name] = "missing"
      continue
    ver = getattr(module, "__version__", None)
    if not ver:
      try:
        import pkg_resources
        ver = pkg_resources.get_distribution(name).version
      except Exception:  # noqa: BLE001
        ver = "unknown"
    versions[name] = ver
  return env, model, versions


def variant_name(args):
  parts = []
  if args.yaw_anchor:
    parts.append("yawanchor")
  if args.reward_v2:
    parts.append("rewardv2")
  if args.random_phase:
    parts.append("randphase")
  if args.start_phases:
    parts.append("phasegrid")
  if args.enable_randomizer:
    parts.append("randenv")
  if args.upstream_init:
    parts.append("upstreaminit")
  return "+".join(parts) if parts else "baseline"


def run(argv=None):
  args = parse_args(argv)
  motion_files, model_file = resolve_paths(args)
  motion_file = motion_files[0]
  clip_change = parse_clip_change(args.clip_change)
  tag = args.tag or os.path.splitext(os.path.basename(motion_file))[0]
  variant = variant_name(args)

  env, model, versions = build_env(args.code_dir, motion_files, model_file,
                                   enable_randomizer=args.enable_randomizer)
  randomizers = find_env_randomizers(env) if args.enable_randomizer else []
  task = env_patch.imitation_task(env)
  if not args.upstream_init:
    # Controlled mode: always start from the reference state so the only
    # difference between variants is the variant itself.
    task._ref_state_init_prob = 1.0

  if args.yaw_anchor:
    env_patch.enable_cycle_heading_anchor(env)
  if args.reward_v2:
    env_patch.apply_velocity_reward_v2(env)
  if clip_change is not None:
    env_patch.enable_clip_change(env, clip_change[0], clip_change[1])

  env_patch.attach_term_logger(env)

  if args.start_phases:
    phases = [float(x) for x in args.start_phases.split(",")]
  else:
    phases = None

  # The environment's own phase sampler draws from an RNG whose state cannot be
  # pinned from the outside, which made "random phase" runs irreproducible
  # across processes. In controlled mode we disable it and drive the phase
  # explicitly instead:
  #   * --start-phases: a fixed grid, cycled over episodes;
  #   * --random-phase: one draw per (seed, episode) from a seeded RandomState.
  # With --upstream-init the upstream sampler is left untouched.
  if not args.upstream_init:
    task._enable_rand_init_time = False

  pyb = env._pybullet_client
  ref_model = task._ref_model
  env_dt = env.env_time_step

  weights = dict(env_patch.REWARD_WEIGHTS_V1)
  if args.reward_v2:
    weights = dict(env_patch.REWARD_WEIGHTS_V2)

  out_dir = os.path.join(args.out, "%s__%s" % (tag, variant))
  os.makedirs(out_dir, exist_ok=True)

  episodes = []
  ep = 0
  for seed in range(args.seeds):
    env.seed(seed)
    for randomizer in randomizers:
      if hasattr(randomizer, "randomization_seed"):
        randomizer.randomization_seed = int(seed)
    # The task holds its own reference RNG (initial state / start time). env.seed
    # does not reach it, so pin it per seed; otherwise --upstream-init draws from
    # the process-global RNG and no run is reproducible.
    env_patch.pin_task_random(task, seed)
    for k in range(args.episodes):
      if args.upstream_init:
        phase = None
      elif phases is not None:
        phase = phases[k % len(phases)]
        env_patch.set_start_phase(env, task, phase)
      elif args.random_phase:
        phase = float(np.random.RandomState(1000 * seed + k).uniform())
        env_patch.set_start_phase(env, task, phase)
      else:
        env_patch.clear_start_phase(task)
        phase = None
      o = env.reset()
      # A randomizer may change the control step; every speed, drift and
      # episode-duration number must use the step actually in force.
      dt = float(env.env_time_step)
      if args.yaw_anchor:
        # _reset_ref_motion installs the opening offset without going through
        # _sync_ref_origin, so the tracker must be re-baselined here.
        env_patch.reset_anchor_tracker(env)
      rewards, terms, pose_errs, root_errs = [], [], [], []
      speeds, speeds_3d, sim_z = [], [], []
      yaw_errs, yaw_signed = [], []
      yaw_fixed_errs, yaw_fixed_signed, anchor_cums = [], [], []
      cum_gaps = []
      gap_yaw, gap_pos = [], []
      rows = []
      joint_rows = []
      nonfoot_first_step = None
      prev_pos = None
      steps = 0
      done = False
      terminated = False
      t0 = time.time()
      while steps < args.max_steps:
        a, _ = model.predict(o, deterministic=True)
        o, r, done, info = env.step(a)
        steps += 1
        # Underlying episode end (fall / task failure), before the wrapper ORs
        # in its own step cap; used to label the termination reason.
        terminated = bool(info.get("terminated", False))
        rewards.append(float(r))

        step_terms = dict(task._last_terms)
        terms.append(step_terms)

        # Tracking metrics are taken from the snapshot captured *inside*
        # task.reward(). Upstream step order is
        #   robot.Step -> reward() -> task.update() -> _update_ref_state()
        # so anything read after env.step() compares the robot against frame
        # t+1 while the reward it earned was measured against frame t.
        snap = task._last_snapshot
        pose_errs.append(metrics.pose_tracking_error(snap["ref_joints"],
                                                     snap["sim_joints"]))

        sim_pos = snap["sim_pos"]
        ref_pos = snap["ref_pos"]
        root_errs.append(metrics.root_tracking_error(sim_pos, ref_pos))
        sim_z.append(float(sim_pos[2]))
        if prev_pos is None:
          delta = np.zeros(3)
        else:
          delta = sim_pos - prev_pos
        speeds_3d.append(float(np.linalg.norm(delta) / dt))
        # Horizontal speed is the gait speed the reference asks for; the 3-D
        # norm additionally counts the vertical bobbing of the torso.
        speeds.append(float(np.linalg.norm(delta[:2]) / dt))
        prev_pos = sim_pos

        signed = metrics.yaw_tracking_error(snap["sim_orn"], snap["ref_orn"])
        yaw_ref = metrics.quat_to_yaw(snap["ref_orn"])
        yaw_signed.append(signed)
        yaw_errs.append(abs(signed))

        # Fixed-reference-frame counterpart: subtract the world-yaw offset our
        # cycle re-anchoring has accumulated (0.0 without --yaw-anchor, where
        # the two series are therefore identical by construction).
        cum = float(snap.get("anchor_cum_yaw", 0.0))
        # Fixed-frame counterpart, built with the quaternion rather than by
        # adding two yaw angles: the opening offset is the only rotation the
        # un-anchored run ever applies, so composing it with the *raw* clip
        # rotation is exactly the reference upstream would track. On a body
        # with pitch or roll, yaw(offset * raw) != yaw(offset) + yaw(raw),
        # which is why the earlier angle-sum version disagreed by up to
        # ~0.12 rad at some frames.
        theta_cf = float(snap.get("anchor_theta_cf_yaw", 0.0))
        ref_orn_fixed = metrics.with_yaw_offset(snap["ref_orn_raw"], theta_cf)
        fixed = metrics.yaw_tracking_error(snap["sim_orn"], ref_orn_fixed)
        # Direct measurement of the accumulated frame rotation: how far the
        # anchored reference has been turned away from the fixed one.
        cum_meas = metrics.wrap_angle(yaw_ref
                                      - metrics.quat_to_yaw(ref_orn_fixed))
        cum_gaps.append(metrics.wrap_angle(cum - cum_meas))
        anchor_cums.append(cum)
        yaw_fixed_signed.append(fixed)
        yaw_fixed_errs.append(abs(fixed))

        # Diagnostic only: how far the reference moved during task.update().
        # This is the one-step offset an earlier version measured by accident;
        # reporting it keeps the size of that misalignment auditable.
        post_ref = pyb.getBasePositionAndOrientation(ref_model)
        gap_yaw.append(abs(metrics.wrap_angle(
            metrics.quat_to_yaw(post_ref[1]) - yaw_ref)))
        gap_pos.append(float(np.linalg.norm(np.array(post_ref[0]) - ref_pos)))

        diag = None
        if args.csv and args.diag:
          diag = termination_state(pyb, env, task, snap)
          lin_vel = np.asarray(snap.get("sim_lin_vel", np.zeros(3)), dtype=float)
          ang_vel = np.asarray(snap.get("sim_ang_vel", np.zeros(3)), dtype=float)
          diag.update({
              "sim_z": float(snap["sim_pos"][2]),
              "lin_vx": float(lin_vel[0]), "lin_vy": float(lin_vel[1]),
              "lin_vz": float(lin_vel[2]),
              "ang_vx": float(ang_vel[0]), "ang_vy": float(ang_vel[1]),
              "ang_vz": float(ang_vel[2]),
              "dt": dt,
          })
          if diag["nonfoot_contacts"] > 0 and nonfoot_first_step is None:
            nonfoot_first_step = steps
          ref_j = np.asarray(snap["ref_joints"], dtype=float)
          sim_j = np.asarray(snap["sim_joints"], dtype=float)
          ref_v = np.asarray(snap.get("ref_joints_vel", np.zeros_like(ref_j)),
                             dtype=float)
          sim_v = np.asarray(snap.get("sim_joints_vel", np.zeros_like(sim_j)),
                             dtype=float)
          for j in range(len(ref_j)):
            joint_rows.append([
                ep, steps, steps * dt, snap["ref_time"], j,
                float(ref_j[j]), float(sim_j[j]),
                float(abs(ref_j[j] - sim_j[j])),
                float(ref_v[j]) if j < len(ref_v) else float("nan"),
                float(sim_v[j]) if j < len(sim_v) else float("nan"),
                (float(abs(ref_v[j] - sim_v[j]))
                 if j < len(ref_v) and j < len(sim_v) else float("nan"))])

        if args.csv:
          rows.append([ep, steps, steps * dt, snap["ref_time"], r]
                      + list(step_terms.values())
                      + [pose_errs[-1], root_errs[-1], yaw_errs[-1],
                         yaw_signed[-1], speeds[-1], speeds_3d[-1],
                         gap_yaw[-1], gap_pos[-1],
                         anchor_cums[-1], yaw_fixed_errs[-1],
                         yaw_fixed_signed[-1], snap.get("ref_yaw_raw",
                                                        float("nan"))]
                      + ([diag["roll_rad"], diag["pitch_rad"],
                          diag["lin_vx"], diag["lin_vy"], diag["lin_vz"],
                          diag["ang_vx"], diag["ang_vy"], diag["ang_vz"],
                          diag["foot_contacts"], diag["nonfoot_contacts"],
                          diag["root_dist_m"], diag["root_rot_angle_rad"],
                          diag["motion_over"], diag["root_pos_fail"],
                          diag["root_rot_fail"], diag["contact_fall"],
                          diag["sim_z"], diag["dt"]] if diag else []))

        if done:
          break

      final_state = termination_state(pyb, env, task, task._last_snapshot)
      termination_causes = sorted(k for k in ("motion_over", "contact_fall",
                                              "root_pos_fail", "root_rot_fail")
                                  if final_state[k]) if terminated else []
      term_means = {k: float(np.mean([t[k] for t in terms]))
                    for k in metrics.TERM_NAMES}
      reward_from_terms = metrics.normalized_return(term_means, weights)
      unwrapped = metrics.unwrap_angles(yaw_signed)
      summary = {
          "episode": ep, "seed": seed, "steps": steps,
          "start_phase": phase,
          "episode_seconds": steps * dt,
          "mean_reward": float(np.mean(rewards)),
          "sum_reward": float(np.sum(rewards)),
          "reward_from_terms": reward_from_terms,
          "reward_matches_terms": bool(abs(reward_from_terms
                                           - float(np.mean(rewards))) < 1e-6),
          "mean_pose_err_rad": float(np.mean(pose_errs)),
          "mean_root_err_m": float(np.mean(root_errs)),
          "max_root_err_m": float(np.max(root_errs)),
          "mean_yaw_err_rad": float(np.mean(yaw_errs)),
          "max_yaw_err_rad": float(np.max(yaw_errs)),
          "final_yaw_err_rad": float(yaw_errs[-1]),
          "final_yaw_err_signed_rad": float(yaw_signed[-1]),
          "yaw_drift_rad_per_s": metrics.drift_rate(unwrapped, dt),
          # same trajectory, scored against the un-re-anchored frame
          "mean_yaw_err_fixed_rad": float(np.mean(yaw_fixed_errs)),
          "max_yaw_err_fixed_rad": float(np.max(yaw_fixed_errs)),
          "final_yaw_err_fixed_rad": float(yaw_fixed_errs[-1]),
          "yaw_drift_fixed_rad_per_s": metrics.drift_rate(
              metrics.unwrap_angles(yaw_fixed_signed), dt),
          "final_anchor_cum_yaw_rad": float(anchor_cums[-1]),
          "max_abs_anchor_cum_yaw_rad": float(np.max(np.abs(anchor_cums))),
          "max_abs_cum_tracker_vs_geometry_rad": (
              float(np.max(np.abs(cum_gaps))) if cum_gaps else 0.0),
          "anchor_cycle_events": (task._anchor_tracker.cycle_events
                                   if args.yaw_anchor else 0),
          "anchor_clip_syncs": (task._anchor_tracker.clip_events
                                if args.yaw_anchor
                                and task._anchor_tracker is not None else 0),
          "mean_speed_mps": float(np.mean(speeds)),
          "mean_speed_3d_mps": float(np.mean(speeds_3d)),
          "mean_sim_z_m": float(np.mean(sim_z)),
          "window5s_mean_reward": metrics.window_mean(rewards, dt, 5.0),
          "env_time_step_s": dt,
          "randomizer_seed": int(seed) if args.enable_randomizer else None,
          "randomizer_count": len(randomizers),
          "randomizer_params": (
              {str(k): jsonable(v)
               for k, v in randomizers[0].get_randomization_parameters().items()}
              if randomizers and hasattr(randomizers[0],
                                         "get_randomization_parameters")
              else None),
          "first_nonfoot_contact_step": nonfoot_first_step,
          "termination": metrics.termination_reason(terminated, done, steps,
                                                    args.max_steps),
          "termination_causes": termination_causes,
          "final_root_dist_m": final_state["root_dist_m"],
          "final_root_rot_angle_rad": final_state["root_rot_angle_rad"],
          "survived": bool(steps >= args.max_steps and not terminated),
          # diagnostic: size of the one-step offset between the reward time
          # point and a post-step read of the reference state
          "mean_offstep_yaw_gap_rad": float(np.mean(gap_yaw)),
          "mean_offstep_ref_pos_gap_m": float(np.mean(gap_pos)),
          "wall_time_s": time.time() - t0,
      }
      for k in metrics.TERM_NAMES:
        summary["term_" + k] = term_means[k]
      episodes.append(summary)

      if args.csv:
        import csv as _csv
        path = os.path.join(out_dir, "ep%02d.csv" % ep)
        with open(path, "w", newline="") as fh:
          w = _csv.writer(fh)
          w.writerow(["episode", "step", "time", "ref_time", "reward"]
                     + ["term_" + k for k in metrics.TERM_NAMES]
                     + ["pose_err_rad", "root_err_m", "yaw_err_rad",
                        "yaw_err_signed_rad", "speed_mps", "speed_3d_mps",
                        "offstep_yaw_gap_rad", "offstep_ref_pos_gap_m",
                        "anchor_cum_yaw_rad", "yaw_err_fixed_rad",
                        "yaw_err_fixed_signed_rad", "ref_yaw_raw_rad"]
                     + (["roll_rad", "pitch_rad",
                         "lin_vx_mps", "lin_vy_mps", "lin_vz_mps",
                         "ang_vx_radps", "ang_vy_radps", "ang_vz_radps",
                         "foot_contacts", "nonfoot_contacts",
                         "root_dist_m", "root_rot_angle_rad",
                         "motion_over", "root_pos_fail", "root_rot_fail",
                         "contact_fall", "sim_z_m", "dt_s"]
                        if args.diag else []))
          w.writerows(rows)
        if args.diag:
          joints_path = os.path.join(out_dir, "ep%02d_joints.csv" % ep)
          with open(joints_path, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["episode", "step", "time", "ref_time", "joint",
                        "ref_joint_rad", "sim_joint_rad", "pose_err_rad",
                        "ref_joint_vel_radps", "sim_joint_vel_radps",
                        "vel_err_radps"])
            w.writerows(joint_rows)

      if not args.quiet:
        print("[eval] %-9s %-22s seed=%d ep=%d steps=%4d reward=%.4f "
              "pose=%.4f root=%.4f yaw=%.4f drift=%.4f rad/s %s"
              % (tag, variant, seed, ep, steps, summary["mean_reward"],
                 summary["mean_pose_err_rad"], summary["mean_root_err_m"],
                 summary["mean_yaw_err_rad"], summary["yaw_drift_rad_per_s"],
                 summary["termination"]))
        if args.yaw_anchor:
          print("           fixed-frame yaw=%.4f rad (anchored %.4f), cum offset %.3f rad,"
                " %d clip sync(s)"
                % (summary["mean_yaw_err_fixed_rad"],
                   summary["mean_yaw_err_rad"],
                   summary["final_anchor_cum_yaw_rad"],
                   summary["anchor_clip_syncs"]))
      ep += 1

  agg = metrics.summarize_episodes(episodes)
  by_seed = metrics.aggregate_by_seed(episodes, keys=[
      "mean_reward", "mean_pose_err_rad", "mean_root_err_m",
      "mean_yaw_err_rad", "yaw_drift_rad_per_s", "mean_speed_mps",
      "mean_yaw_err_fixed_rad", "yaw_drift_fixed_rad_per_s"])
  payload = {
      "tag": tag, "variant": variant,
      "run_id": args.run_id,
      "motion_file": motion_file, "model_file": model_file,
      "motion_files": list(motion_files),
      "clip_change_s": list(clip_change) if clip_change else None,
      "model_sha256": sha256(model_file),
      "env_time_step": env_dt,
      "provenance": {"upstream_commit": UPSTREAM_COMMIT,
                     "versions": versions},
      "protocol": {
          "episodes": args.episodes, "seeds": args.seeds,
          "seed_meaning": "initial conditions (not training seeds)",
          "init_mode": "upstream" if args.upstream_init else "controlled",
          "random_phase": bool(args.random_phase),
          "enable_randomizer": bool(args.enable_randomizer),
          "randomizer_count": len(randomizers),
          "randomizer_classes": [type(r).__name__ for r in randomizers],
          "randomizer_seed": ("episode seed, applied as"
                              " randomizer.randomization_seed before each reset"
                              if args.enable_randomizer else None),
          "start_phases": phases,
          "yaw_anchor": bool(args.yaw_anchor),
          "heading_frame": ("post-hoc fixed-frame reconstruction from the same"
                             " trajectory (ref_yaw - cum_cycle_anchor_yaw); not"
                             " the error a policy would produce with anchoring"
                             " disabled"
                             if args.yaw_anchor else "not applicable (no anchoring)"),
          "reward_v2": bool(args.reward_v2),
          "metric_timepoint": "inside task.reward() (reference frame t, "
                              "before task.update())",
          "task_rng": "np.random.RandomState(seed) per seed (pinned by us; "
                      "upstream leaves this RNG unseeded)",
          "weights": weights,
          "max_steps": args.max_steps,
      },
      "aggregate": agg,
      "by_seed": by_seed,
      "episodes": episodes,
  }
  with open(os.path.join(out_dir, "summary.json"), "w") as fh:
    json.dump(payload, fh, indent=2)

  env.close()
  if not args.quiet:
    print("[eval] %s -> %s" % (tag, out_dir))
    print("[eval] mean reward %.4f +- %.4f | pose %.4f rad | root %.4f m | "
          "yaw %.4f rad | drift %.4f rad/s | speed(horiz) %.4f m/s"
          % (agg.get("mean_reward_mean", float("nan")),
             agg.get("mean_reward_std", 0.0),
             agg.get("mean_pose_err_rad_mean", float("nan")),
             agg.get("mean_root_err_m_mean", float("nan")),
             agg.get("mean_yaw_err_rad_mean", float("nan")),
             agg.get("yaw_drift_rad_per_s_mean", float("nan")),
             agg.get("mean_speed_mps_mean", float("nan"))))
  return payload


if __name__ == "__main__":
  run()
