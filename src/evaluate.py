"""Paper-grade evaluation for motion-imitation policies.

Differences from a plain rollout loop (see docs/IMPROVEMENTS.md):

* multi-seed, multi-episode runs with mean/std instead of a single deterministic
  episode, optionally starting from random points along the reference clip
  (the protocol the paper uses for its Table V numbers);
* the five reward terms are logged separately and ``mean(reward)`` is
  cross-checked against the paper-style normalised return reconstructed from
  those terms;
* unbounded drift is separated from bounded error by fitting the slope of the
  per-step heading error, and a trailing-window reward removes the
  accumulation artefact from the reported score.

Usage (from the repository root, inside the project's conda env):

  python -m src.evaluate --motion dog_pace --policy dog_pace \
      --episodes 5 --seeds 3 --random-phase --yaw-anchor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CODE_DIR = os.path.join(REPO_ROOT, "third_party", "motion_imitation")
MOTION_DIR = os.path.join(DEFAULT_CODE_DIR, "data", "motions")
POLICY_DIR = os.path.join(DEFAULT_CODE_DIR, "data", "policies")

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
  p.add_argument("--seeds", type=int, default=1)
  p.add_argument("--random-phase", action="store_true",
                 help="start each episode from a random point in the clip")
  p.add_argument("--start-phases", default=None,
                 help="comma separated phases in [0,1), overrides random phase")
  p.add_argument("--yaw-anchor", action="store_true",
                 help="improvement 1: re-anchor reference heading every cycle")
  p.add_argument("--reward-v2", action="store_true",
                 help="improvement 2: rebalanced velocity term and weights")
  p.add_argument("--max-steps", type=int, default=600)
  p.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "eval"))
  p.add_argument("--tag", default=None)
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


def build_env(code_dir, motion_file, model_file):
  sys.path.insert(0, os.path.abspath(code_dir))
  # Single-threaded math kernels: with the default thread pool the same policy
  # and the same seed can diverge across processes at the 1e-7 level, which is
  # enough for a marginally stable gait to roll out differently over 600 steps.
  # Pinning the pool makes runs bit-reproducible across processes.
  for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
    os.environ[var] = "1"
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

  import tensorflow as tf  # noqa: F401
  import envs.env_builder as env_builder
  import learning.imitation_policies as imitation_policies
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
  return env, model


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
  return "+".join(parts) if parts else "baseline"


def run(argv=None):
  args = parse_args(argv)
  motion_file, model_file = resolve_paths(args)
  tag = args.tag or os.path.splitext(os.path.basename(motion_file))[0]
  variant = variant_name(args)

  env, model = build_env(args.code_dir, motion_file, model_file)
  task = env_patch.imitation_task(env)
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
  # across processes. Disable it and drive the phase explicitly instead:
  #   * --start-phases: a fixed grid, cycled over episodes;
  #   * --random-phase: one draw per (seed, episode) from a seeded RandomState,
  #     i.e. the same distribution, but reproducible.
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
      if phases is not None:
        phase = phases[k % len(phases)]
        env_patch.set_start_phase(env, task, phase)
      elif args.random_phase:
        phase = float(np.random.RandomState(1000 * seed + k).uniform())
        env_patch.set_start_phase(env, task, phase)
      else:
        env_patch.clear_start_phase(task)
        phase = None
      o = env.reset()
      rewards, terms, pose_errs, root_errs, yaw_errs, speeds, sim_z = (
          [], [], [], [], [], [], [])
      rows = []
      prev_pos = np.array(pyb.getBasePositionAndOrientation(sim_model)[0])
      steps = 0
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
        speeds.append(float(np.linalg.norm(sim_pos - prev_pos) / env_dt))
        prev_pos = sim_pos

        sim_orn = np.array(pyb.getBasePositionAndOrientation(sim_model)[1])
        ref_orn = np.array(pyb.getBasePositionAndOrientation(ref_model)[1])
        yaw_sim = 2.0 * np.arctan2(sim_orn[2], sim_orn[3])
        yaw_ref = 2.0 * np.arctan2(ref_orn[2], ref_orn[3])
        yaw_errs.append(float(abs((yaw_sim - yaw_ref + np.pi)
                                  % (2 * np.pi) - np.pi)))
        if args.csv:
          rows.append([ep, steps, steps * env_dt, r] + list(step_terms.values())
                      + [pose_errs[-1], root_errs[-1], yaw_errs[-1], speeds[-1]])

        if done:
          break

      term_means = {k: float(np.mean([t[k] for t in terms]))
                    for k in metrics.TERM_NAMES}
      reward_from_terms = metrics.normalized_return(term_means, weights)
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
          "yaw_drift_rad_per_s": metrics.drift_rate(yaw_errs, env_dt),
          "mean_speed_mps": float(np.mean(speeds)),
          "mean_sim_z_m": float(np.mean(sim_z)),
          "window5s_mean_reward": metrics.window_mean(rewards, env_dt, 5.0),
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
                        "speed_mps"])
          w.writerows(rows)

      if not args.quiet:
        print("[eval] %-9s %-22s seed=%d ep=%d steps=%4d reward=%.4f "
              "pose=%.4f root=%.4f yaw=%.4f drift=%.4f rad/s"
              % (tag, variant, seed, ep, steps, summary["mean_reward"],
                 summary["mean_pose_err_rad"], summary["mean_root_err_m"],
                 summary["mean_yaw_err_rad"], summary["yaw_drift_rad_per_s"]))
      ep += 1

  agg = metrics.summarize_episodes(episodes)
  payload = {
      "tag": tag, "variant": variant,
      "motion_file": motion_file, "model_file": model_file,
      "env_time_step": env_dt,
      "protocol": {
          "episodes": args.episodes, "seeds": args.seeds,
          "random_phase": bool(args.random_phase),
          "start_phases": phases,
          "yaw_anchor": bool(args.yaw_anchor),
          "reward_v2": bool(args.reward_v2),
          "weights": weights,
          "max_steps": args.max_steps,
      },
      "aggregate": agg,
  }
  with open(os.path.join(out_dir, "summary.json"), "w") as fh:
    json.dump(payload, fh, indent=2)

  env.close()
  if not args.quiet:
    print("[eval] %s -> %s" % (tag, out_dir))
    print("[eval] mean reward %.4f +- %.4f | pose %.4f rad | root %.4f m | "
          "yaw %.4f rad | drift %.4f rad/s"
          % (agg.get("mean_reward_mean", float("nan")),
             agg.get("mean_reward_std", 0.0),
             agg.get("mean_pose_err_rad_mean", float("nan")),
             agg.get("mean_root_err_m_mean", float("nan")),
             agg.get("mean_yaw_err_rad_mean", float("nan")),
             agg.get("yaw_drift_rad_per_s_mean", float("nan"))))
  return payload


if __name__ == "__main__":
  run()
