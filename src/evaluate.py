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
  p.add_argument("--quiet", action="store_true")
  return p.parse_args(argv)


def resolve_paths(args):
  motion_file = args.motion_file
  if motion_file is None:
    if args.motion is None:
      raise SystemExit("give --motion or --motion_file")
    motion_file = os.path.join(MOTION_DIR, args.motion + ".txt")
  model_file = args.model_file
  policy = args.policy
  if model_file is None:
    if policy is None:
      policy = os.path.splitext(os.path.basename(motion_file))[0]
    model_file = os.path.join(POLICY_DIR, policy + ".zip")
  for path in (motion_file, model_file):
    if not os.path.isfile(path):
      raise SystemExit("missing file: %s" % path)
  return motion_file, model_file


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def build_env(code_dir, motion_file, model_file):
  sys.path.insert(0, os.path.abspath(code_dir))
  import tensorflow as tf  # noqa: F401
  import envs.env_builder as env_builder
  import learning.imitation_policies as imitation_policies
  # note: ppo_imitation pulls in mpi4py (MPI) even for single-process evaluation
  import learning.ppo_imitation as ppo_imitation

  env = env_builder.build_imitation_env(
      motion_files=[motion_file],
      num_parallel_envs=1,
      mode="test",
      enable_randomizer=False,
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
  if args.upstream_init:
    parts.append("upstreaminit")
  return "+".join(parts) if parts else "baseline"


def run(argv=None):
  args = parse_args(argv)
  motion_file, model_file = resolve_paths(args)
  tag = args.tag or os.path.splitext(os.path.basename(motion_file))[0]
  variant = variant_name(args)

  env, model, versions = build_env(args.code_dir, motion_file, model_file)
  task = env_patch.imitation_task(env)
  if not args.upstream_init:
    # Controlled mode: always start from the reference state so the only
    # difference between variants is the variant itself.
    task._ref_state_init_prob = 1.0

  if args.yaw_anchor:
    env_patch.enable_cycle_heading_anchor(env)
  if args.reward_v2:
    env_patch.apply_velocity_reward_v2(env)

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
  robot = env.robot
  sim_model = robot.quadruped
  ref_model = task._ref_model
  env_dt = env.env_time_step
  n_joints = pyb.getNumJoints(ref_model)

  weights = dict(env_patch.REWARD_WEIGHTS_V1)
  if args.reward_v2:
    weights = dict(env_patch.REWARD_WEIGHTS_V2)

  out_dir = os.path.join(args.out, "%s__%s" % (tag, variant))
  os.makedirs(out_dir, exist_ok=True)

  episodes = []
  ep = 0
  for seed in range(args.seeds):
    env.seed(seed)
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
      rewards, terms, pose_errs, root_errs = [], [], [], []
      speeds, speeds_3d, sim_z = [], [], []
      yaw_errs, yaw_signed = [], []
      rows = []
      prev_pos = np.array(pyb.getBasePositionAndOrientation(sim_model)[0])
      steps = 0
      done = False
      t0 = time.time()
      while steps < args.max_steps:
        a, _ = model.predict(o, deterministic=True)
        o, r, done, info = env.step(a)
        steps += 1
        rewards.append(float(r))

        step_terms = dict(task._last_terms)
        terms.append(step_terms)

        motion = task.get_active_motion()
        ref_pose = task._ref_pose
        ref_joints = np.array(motion.get_frame_joints(ref_pose))
        sim_joints = []
        for j in range(n_joints):
          jq = pyb.getJointStateMultiDof(sim_model, j)[0]
          if len(jq) > 0:
            sim_joints.append(float(jq[0]))
        sim_joints = np.array(sim_joints)
        pose_errs.append(float(np.mean(np.abs(ref_joints - sim_joints))))

        sim_pos = np.array(pyb.getBasePositionAndOrientation(sim_model)[0])
        ref_pos = np.array(pyb.getBasePositionAndOrientation(ref_model)[0])
        root_errs.append(float(np.linalg.norm(sim_pos - ref_pos)))
        sim_z.append(float(sim_pos[2]))
        delta = sim_pos - prev_pos
        speeds_3d.append(float(np.linalg.norm(delta) / env_dt))
        # Horizontal speed is the gait speed the reference asks for; the 3-D
        # norm additionally counts the vertical bobbing of the torso.
        speeds.append(float(np.linalg.norm(delta[:2]) / env_dt))
        prev_pos = sim_pos

        sim_orn = np.array(pyb.getBasePositionAndOrientation(sim_model)[1])
        ref_orn = np.array(pyb.getBasePositionAndOrientation(ref_model)[1])
        yaw_sim = metrics.quat_to_yaw(sim_orn)
        yaw_ref = metrics.quat_to_yaw(ref_orn)
        signed = metrics.wrap_angle(yaw_sim - yaw_ref)
        yaw_signed.append(signed)
        yaw_errs.append(abs(signed))
        if args.csv:
          rows.append([ep, steps, steps * env_dt, r] + list(step_terms.values())
                      + [pose_errs[-1], root_errs[-1], yaw_errs[-1],
                         yaw_signed[-1], speeds[-1], speeds_3d[-1]])

        if done:
          break

      term_means = {k: float(np.mean([t[k] for t in terms]))
                    for k in metrics.TERM_NAMES}
      reward_from_terms = metrics.normalized_return(term_means, weights)
      unwrapped = metrics.unwrap_angles(yaw_signed)
      summary = {
          "episode": ep, "seed": seed, "steps": steps,
          "start_phase": phase,
          "episode_seconds": steps * env_dt,
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
          "yaw_drift_rad_per_s": metrics.drift_rate(unwrapped, env_dt),
          "mean_speed_mps": float(np.mean(speeds)),
          "mean_speed_3d_mps": float(np.mean(speeds_3d)),
          "mean_sim_z_m": float(np.mean(sim_z)),
          "window5s_mean_reward": metrics.window_mean(rewards, env_dt, 5.0),
          "termination": metrics.termination_reason(done, steps,
                                                    args.max_steps),
          "survived": bool(steps >= args.max_steps),
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
          w.writerow(["episode", "step", "time", "reward"]
                     + ["term_" + k for k in metrics.TERM_NAMES]
                     + ["pose_err_rad", "root_err_m", "yaw_err_rad",
                        "yaw_err_signed_rad", "speed_mps", "speed_3d_mps"])
          w.writerows(rows)

      if not args.quiet:
        print("[eval] %-9s %-22s seed=%d ep=%d steps=%4d reward=%.4f "
              "pose=%.4f root=%.4f yaw=%.4f drift=%.4f rad/s %s"
              % (tag, variant, seed, ep, steps, summary["mean_reward"],
                 summary["mean_pose_err_rad"], summary["mean_root_err_m"],
                 summary["mean_yaw_err_rad"], summary["yaw_drift_rad_per_s"],
                 summary["termination"]))
      ep += 1

  agg = metrics.summarize_episodes(episodes)
  by_seed = metrics.aggregate_by_seed(episodes, keys=[
      "mean_reward", "mean_pose_err_rad", "mean_root_err_m",
      "mean_yaw_err_rad", "yaw_drift_rad_per_s", "mean_speed_mps"])
  payload = {
      "tag": tag, "variant": variant,
      "run_id": args.run_id,
      "motion_file": motion_file, "model_file": model_file,
      "model_sha256": sha256(model_file),
      "env_time_step": env_dt,
      "provenance": {"upstream_commit": UPSTREAM_COMMIT,
                     "versions": versions},
      "protocol": {
          "episodes": args.episodes, "seeds": args.seeds,
          "seed_meaning": "initial conditions (not training seeds)",
          "init_mode": "upstream" if args.upstream_init else "controlled",
          "random_phase": bool(args.random_phase),
          "start_phases": phases,
          "yaw_anchor": bool(args.yaw_anchor),
          "reward_v2": bool(args.reward_v2),
          "weights": weights,
          "max_steps": args.max_steps,
      },
      "aggregate": agg,
      "by_seed": by_seed,
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
