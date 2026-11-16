# coding=utf-8
"""Variant-aware PPO training entry point for controlled ablations.

The upstream ``third_party/motion_imitation/run.py`` has no way to install the
improvement patches, so training two "arms" with it would silently run the
same condition twice. This entry point mirrors upstream environment
construction and PPO hyper-parameters, then installs exactly one patch before
the first rollout:

  baseline    upstream task untouched
  yaw_anchor  env_patch.enable_cycle_heading_anchor
  reward_v2   env_patch.apply_velocity_reward_v2

``--total-timesteps`` is the **aggregate** step budget: PPOImitation.learn
all-reduces ``total_timestep`` across MPI ranks and compares the running sum
with this value (``learning/ppo_imitation.py:370-379``).

Usage (repo root):
  mpiexec -n 8 python -m src.train_ppo --motion dog_pace --variant baseline \
      --seed 0 --total-timesteps 300000 --output-dir outputs/... --run-id ...
"""

from __future__ import absolute_import, division, print_function

import os

# Thread limits must be set before NumPy / TensorFlow import.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import csv
import datetime
import hashlib
import json
import random
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_DIR = os.path.join(REPO_ROOT, "third_party", "motion_imitation")
for _path in (REPO_ROOT, UPSTREAM_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mpi4py import MPI  # noqa: E402
import tensorflow as tf  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.imitation_policies as imitation_policies  # noqa: E402
import learning.ppo_imitation as ppo_imitation  # noqa: E402
from src import env_patch  # noqa: E402

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256
GAMMA = 0.95
CLIP_PARAM = 0.2
OPTIM_EPOCHS = 1
OPTIM_STEPSIZE = 1e-5
LAMBDA = 0.95
ADAM_EPSILON = 1e-5


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args):
    try:
        out = subprocess.check_output(["git"] + list(args), cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return "unknown"


class EpisodeLogger(object):
    """Rank-0 episode CSV so the smoke run has a learning curve to inspect."""

    def __init__(self, env, csv_path, is_root):
        self._env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._is_root = is_root
        self._fh = open(csv_path, "w", newline="") if is_root else None
        self._writer = csv.writer(self._fh) if is_root else None
        if is_root:
            self._writer.writerow(["episode", "global_step", "episode_return",
                                   "episode_length", "wall_time_s"])
        self._episode = 0
        self._ret = 0.0
        self._len = 0
        self.global_step = 0
        self._t0 = time.time()

    def __getattr__(self, attr):
        return getattr(self._env, attr)

    def reset(self, **kwargs):
        self._ret = 0.0
        self._len = 0
        return self._env.reset(**kwargs)

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        self.global_step += 1
        self._ret += rew
        self._len += 1
        if done:
            self._episode += 1
            if self._is_root:
                self._writer.writerow([self._episode, self.global_step,
                                       self._ret, self._len,
                                       time.time() - self._t0])
                self._fh.flush()
            self._ret = 0.0
            self._len = 0
        return obs, rew, done, info

    def close(self):
        if self._fh is not None:
            self._fh.close()
        self._env.close()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Train PPO imitation with one explicit ablation variant.")
    ap.add_argument("--motion", default="dog_pace")
    ap.add_argument("--motion-file", default=None)
    ap.add_argument("--variant", required=True,
                    choices=("baseline", "yaw_anchor", "reward_v2"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--total-timesteps", type=int, default=300000,
                    help="Aggregate timesteps across MPI ranks.")
    ap.add_argument("--output-dir", default=None,
                    help="Required unless --dry-run; unique directory per arm.")
    ap.add_argument("--run-id", default=None,
                    help="Required unless --dry-run; independent run identifier.")
    ap.add_argument("--expected-procs", type=int, default=8)
    ap.add_argument("--disable-randomizer", action="store_true")
    ap.add_argument("--int-save-freq", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def set_rank_seed(base_seed, rank):
    rank_seed = int(base_seed + 97 * rank)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    tf.set_random_seed(rank_seed)
    return rank_seed


def resolve_motion_file(args):
    path = args.motion_file or os.path.join(
        UPSTREAM_DIR, "data", "motions", args.motion + ".txt")
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Motion file not found: " + path)
    return path


def install_variant(env, variant):
    task = env_patch.imitation_task(env)
    if variant == "baseline":
        pass
    elif variant == "yaw_anchor":
        env_patch.enable_cycle_heading_anchor(env)
        if not getattr(task, "_heading_anchor_installed", False):
            raise RuntimeError("yaw_anchor requested but patch is not installed")
    elif variant == "reward_v2":
        env_patch.apply_velocity_reward_v2(env)
        if not getattr(task, "_velocity_reward_v2", False):
            raise RuntimeError("reward_v2 requested but patch is not installed")
    return task


def patch_flags(task):
    return {
        "heading_anchor_installed": bool(
            getattr(task, "_heading_anchor_installed", False)),
        "velocity_reward_v2": bool(getattr(task, "_velocity_reward_v2", False)),
        "pose_weight": float(getattr(task, "_pose_weight", 0.5)),
        "velocity_weight": float(getattr(task, "_velocity_weight", 0.05)),
    }


def build_manifest(args, motion_file, rank_seed, num_procs, task):
    dirty = bool(git_text("status", "--porcelain"))
    try:
        motion_sha = sha256_file(motion_file)
    except Exception:
        motion_sha = "unavailable"
    return {
        "run_id": args.run_id,
        "variant": args.variant,
        "patch_flags": patch_flags(task),
        "motion_file": motion_file,
        "motion_sha256": motion_sha,
        "code": {
            "git_commit": git_text("rev-parse", "HEAD"),
            "git_branch": git_text("branch", "--show-current"),
            "git_describe": git_text("describe", "--tags", "--always"),
            "working_tree_dirty": dirty,
            "vendored_upstream_commit": "2f206ab",
        },
        "training": {
            "base_seed": int(args.seed),
            "rank_seed_rule": "base_seed + 97 * MPI_rank",
            "rank_0_seed": int(args.seed),
            "current_rank_seed": int(rank_seed),
            "mpi_processes": int(num_procs),
            "requested_aggregate_timesteps": int(args.total_timesteps),
            "timestep_unit": "aggregate across MPI ranks (allreduce)",
            "environment_randomizer": not bool(args.disable_randomizer),
            "curriculum_episode_length_start": 20,
            "curriculum_episode_length_end": 600,
            "curriculum_steps_global": 30000000,
        },
        "ppo": {
            "gamma": GAMMA,
            "timesteps_per_actorbatch_global": TIMESTEPS_PER_ACTORBATCH,
            "optim_epochs": OPTIM_EPOCHS,
            "optim_stepsize": OPTIM_STEPSIZE,
            "optim_batchsize_global": OPTIM_BATCHSIZE,
            "clip_param": CLIP_PARAM,
            "lambda": LAMBDA,
            "adam_epsilon": ADAM_EPSILON,
            "policy": "ImitationPolicy",
            "network": {"pi": [512, 256], "vf": [512, 256]},
        },
        "runtime": {
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "started_at_utc": utc_now(),
        },
    }


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if num_procs != args.expected_procs:
        if rank == 0:
            print("ERROR: expected {} MPI processes, got {}.".format(
                args.expected_procs, num_procs), file=sys.stderr)
        raise SystemExit(2)

    if not args.dry_run and (not args.output_dir or not args.run_id):
        raise SystemExit("--output-dir and --run-id are required unless --dry-run")

    rank_seed = set_rank_seed(args.seed, rank)
    motion_file = resolve_motion_file(args)
    output_dir = (os.path.abspath(args.output_dir)
                  if args.output_dir else None)

    env = env_builder.build_imitation_env(
        motion_files=[motion_file],
        num_parallel_envs=num_procs,
        mode="train",
        enable_randomizer=not args.disable_randomizer,
        enable_rendering=False)

    task = install_variant(env, args.variant)
    # The task samples initial state / start time from its own RNG; give it a
    # private, seeded RandomState so both arms share the same randomization
    # schedule. Upstream never seeds this object.
    env_patch.pin_task_random(task, rank_seed)

    if args.dry_run:
        obs = env.reset()
        if not np.all(np.isfinite(obs)):
            raise RuntimeError("dry-run: observation has NaN/Inf")
        if args.variant == "yaw_anchor" and not task._heading_anchor_installed:
            raise RuntimeError("dry-run: yaw-anchor patch inactive")
        if args.variant == "reward_v2" and not task._velocity_reward_v2:
            raise RuntimeError("dry-run: reward-v2 patch inactive")
        print("ENV_READY rank={} variant={} obs={} seed={} flags={}".format(
            rank, args.variant, tuple(obs.shape), rank_seed,
            json.dumps(patch_flags(task), sort_keys=True)))
        env.close()
        return

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        write_json(os.path.join(output_dir, "run_manifest.json"),
                   build_manifest(args, motion_file, rank_seed, num_procs, task))
    comm.Barrier()

    env = EpisodeLogger(env, os.path.join(output_dir, "episodes.csv"),
                        is_root=(rank == 0))

    policy_kwargs = {"net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
                     "act_fun": tf.nn.relu}
    model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=GAMMA,
        timesteps_per_actorbatch=int(np.ceil(
            float(TIMESTEPS_PER_ACTORBATCH) / num_procs)),
        clip_param=CLIP_PARAM,
        optim_epochs=OPTIM_EPOCHS,
        optim_stepsize=OPTIM_STEPSIZE,
        optim_batchsize=int(np.ceil(float(OPTIM_BATCHSIZE) / num_procs)),
        lam=LAMBDA,
        adam_epsilon=ADAM_EPSILON,
        schedule="constant",
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir if rank == 0 else None,
        verbose=1 if rank == 0 else 0,
        seed=rank_seed,
        n_cpu_tf_sess=1)

    callback = None
    if args.int_save_freq > 0:
        from stable_baselines.common.callbacks import CheckpointCallback
        callback = [CheckpointCallback(
            save_freq=args.int_save_freq,
            save_path=os.path.join(output_dir, "intermediate"),
            name_prefix="model")]

    model_path = os.path.join(output_dir, "model.zip")
    t0 = time.time()
    try:
        model.learn(total_timesteps=args.total_timesteps,
                    save_path=model_path if rank == 0 else None,
                    callback=callback)
        comm.Barrier()
        elapsed = time.time() - t0
        if rank == 0:
            steps = int(getattr(model, "num_timesteps", 0))
            summary = {
                "status": "completed",
                "run_id": args.run_id,
                "variant": args.variant,
                "patch_flags": patch_flags(task),
                "requested_aggregate_timesteps": int(args.total_timesteps),
                "actual_aggregate_timesteps": steps,
                "mpi_processes": int(num_procs),
                "base_seed": int(args.seed),
                "wall_time_seconds": float(elapsed),
                "aggregate_steps_per_second": (
                    float(steps) / elapsed if elapsed > 0 else None),
                "rank0_episodes": int(getattr(env, "_episode", 0)),
                "model_file": model_path,
                "model_exists": os.path.isfile(model_path),
                "finished_at_utc": utc_now(),
            }
            write_json(os.path.join(output_dir, "training_summary.json"),
                       summary)
            print("TRAINING_COMPLETE")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        if rank == 0:
            write_json(os.path.join(output_dir, "training_failure.json"), {
                "status": "failed",
                "run_id": args.run_id,
                "variant": args.variant,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "failed_at_utc": utc_now(),
            })
        raise
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            model.sess.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
