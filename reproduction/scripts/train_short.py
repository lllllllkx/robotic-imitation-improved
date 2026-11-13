"""Short PPO imitation-learning run that mirrors motion_imitation/run.py.

The upstream training entry point (run.py) is left untouched; this script
builds the exact same environment/model and adds an episode-level CSV logger
so the learning curve can be plotted and compared with the paper.

Supports MPI (mpiexec -n N), like run.py. Only rank 0 writes logs / models.

Usage example:
  mpiexec -n 4 python scripts/train_short.py \
      --code_dir <...>/code/motion_imitation \
      --motion_file <...>/data/motions/dog_pace.txt \
      --output_dir <...>/outputs/train/dog_pace_400k \
      --total_timesteps 400000 --seed 0
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256


class EpisodeLogger(object):
  """Minimal env wrapper that records per-episode returns and lengths."""

  def __init__(self, env, csv_path, print_every=10, is_root=True):
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
    self._ep_return = 0.0
    self._ep_len = 0
    self.global_step = 0
    self.print_every = print_every
    self.t0 = time.time()
    self._o = None

  def __getattr__(self, attr):
    return getattr(self._env, attr)

  def reset(self, **kwargs):
    self._o = self._env.reset(**kwargs)
    self._ep_return = 0.0
    self._ep_len = 0
    return self._o

  def step(self, action):
    o, r, done, info = self._env.step(action)
    self.global_step += 1
    self._ep_return += r
    self._ep_len += 1
    self._o = o
    if done:
      self._episode += 1
      if self._is_root:
        self._writer.writerow([self._episode, self.global_step,
                               self._ep_return, self._ep_len,
                               time.time() - self.t0])
        if self.print_every and self._episode % self.print_every == 0:
          self._fh.flush()
          print("[train] ep=%d step=%d return=%.2f len=%d" %
                (self._episode, self.global_step, self._ep_return,
                 self._ep_len))
      self._env.reset()
      self._ep_return = 0.0
      self._ep_len = 0
    return o, r, done, info

  def close(self):
    if self._fh is not None:
      self._fh.close()
    self._env.close()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--code_dir", required=True)
  ap.add_argument("--motion_file", required=True)
  ap.add_argument("--output_dir", required=True)
  ap.add_argument("--total_timesteps", type=int, default=200000)
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--int_save_freq", type=int, default=0)
  args = ap.parse_args()

  sys.path.insert(0, os.path.abspath(args.code_dir))
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

  from mpi4py import MPI
  import tensorflow as tf
  import envs.env_builder as env_builder
  import learning.imitation_policies as imitation_policies
  import learning.ppo_imitation as ppo_imitation

  rank = MPI.COMM_WORLD.Get_rank()
  num_procs = MPI.COMM_WORLD.Get_size()
  is_root = rank == 0

  # Same seeding scheme as run.py: seed is shifted per rank so workers explore
  # different rollouts, while rank 0 keeps the user-provided seed.
  seed = args.seed if is_root else args.seed + 97 * rank
  tf.set_random_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  if is_root:
    os.makedirs(args.output_dir, exist_ok=True)

  env = env_builder.build_imitation_env(
      motion_files=[args.motion_file],
      num_parallel_envs=num_procs,
      mode="train",
      enable_randomizer=True,
      enable_rendering=False)
  env = EpisodeLogger(env, os.path.join(args.output_dir, "episodes.csv"),
                      is_root=is_root)

  # run.py divides the per-batch budget across MPI workers.
  timesteps_per_actorbatch = int(np.ceil(float(TIMESTEPS_PER_ACTORBATCH) /
                                         num_procs))
  optim_batchsize = int(np.ceil(float(OPTIM_BATCHSIZE) / num_procs))

  policy_kwargs = {
      "net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
      "act_fun": tf.nn.relu,
  }
  model = ppo_imitation.PPOImitation(
      policy=imitation_policies.ImitationPolicy,
      env=env,
      gamma=0.95,
      timesteps_per_actorbatch=timesteps_per_actorbatch,
      clip_param=0.2,
      optim_epochs=1,
      optim_stepsize=1e-5,
      optim_batchsize=optim_batchsize,
      lam=0.95,
      adam_epsilon=1e-5,
      schedule="constant",
      policy_kwargs=policy_kwargs,
      tensorboard_log=args.output_dir if is_root else None,
      verbose=1 if is_root else 0)

  callback = None
  if args.int_save_freq > 0:
    from stable_baselines.common.callbacks import CheckpointCallback
    callback = [CheckpointCallback(
        save_freq=args.int_save_freq,
        save_path=os.path.join(args.output_dir, "intermediate"),
        name_prefix="model")]

  t0 = time.time()
  model.learn(total_timesteps=args.total_timesteps,
              save_path=os.path.join(args.output_dir, "model.zip") if is_root
              else None,
              callback=callback)
  elapsed = time.time() - t0

  if is_root:
    model.save(os.path.join(args.output_dir, "final_model.zip"))
  env.close()

  if is_root:
    steps_per_sec = args.total_timesteps / max(elapsed, 1e-9)
    meta = {
        "motion_file": args.motion_file,
        "total_timesteps": args.total_timesteps,
        "seed": args.seed,
        "num_procs": num_procs,
        "timesteps_per_actorbatch": timesteps_per_actorbatch,
        "wall_time_s": elapsed,
        "steps_per_sec_total": steps_per_sec,
        "num_episodes_rank0": env._episode,
    }
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
      json.dump(meta, f, indent=2)
    print("[train] finished: %s (%.1f s, %.1f steps/s aggregated)" %
          (args.output_dir, elapsed, steps_per_sec))


if __name__ == "__main__":
  main()
