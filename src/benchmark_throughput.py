# coding=utf-8
"""Measure full-length (600-step) environment throughput with MPI.

The action source is a pretrained policy that actually survives the full
episode; only ``env.step`` wall time is counted as simulation cost. Policy
inference is timed separately and reported, so "simulator speed" and
"policy+simulator speed" are never conflated.

Every episode records the real number of steps and whether it ended by
``terminated`` (fall) or by the 600-step limit, so a ``--max-steps 600`` run
cannot be mistaken for a completed full-length episode.

Usage:
  mpiexec -n 8 python -m src.benchmark_throughput \
      --motion dog_pace --repeats 3 --episodes-per-rank 2 \
      --out results/throughput_v1/env_randomizer_off
"""

from __future__ import absolute_import, division, print_function

import os

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


def utc_now():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Full-length environment throughput benchmark.")
    ap.add_argument("--motion", default="dog_pace")
    ap.add_argument("--policy-file", default=None,
                    help="Pretrained policy that survives 600 steps.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--episodes-per-rank", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--expected-procs", type=int, default=8)
    ap.add_argument("--enable-randomizer", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="r20261116_throughput_env_v1")
    ap.add_argument("--tag", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_procs = comm.Get_size()
    if num_procs != args.expected_procs:
        raise SystemExit("expected %d MPI processes, got %d"
                         % (args.expected_procs, num_procs))

    motion_file = os.path.join(UPSTREAM_DIR, "data", "motions",
                               args.motion + ".txt")
    policy_file = args.policy_file or os.path.join(
        UPSTREAM_DIR, "data", "policies", args.motion + ".zip")
    out_dir = os.path.abspath(args.out)
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)

    env = env_builder.build_imitation_env(
        motion_files=[motion_file],
        num_parallel_envs=num_procs,
        mode="test",  # episode length fixed at 600, no curriculum
        enable_randomizer=args.enable_randomizer,
        enable_rendering=False)

    model = ppo_imitation.PPOImitation(
        policy=imitation_policies.ImitationPolicy,
        env=env,
        gamma=0.95,
        timesteps_per_actorbatch=int(np.ceil(4096.0 / num_procs)),
        clip_param=0.2,
        optim_epochs=1,
        optim_stepsize=1e-5,
        optim_batchsize=int(np.ceil(256.0 / num_procs)),
        lam=0.95,
        adam_epsilon=1e-5,
        schedule="constant",
        policy_kwargs={"net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
                       "act_fun": tf.nn.relu},
        tensorboard_log=None,
        verbose=0)
    model.load_parameters(policy_file)

    rows = []
    repeats = []
    for rep in range(args.repeats):
        comm.Barrier()
        t0 = time.time()
        rep_rows = []
        for ep in range(args.episodes_per_rank):
            obs = env.reset()
            steps = 0
            done = False
            terminated = None
            env_time = 0.0
            policy_time = 0.0
            while not done and steps < args.max_steps:
                ta = time.time()
                action, _ = model.predict(obs, deterministic=True)
                policy_time += time.time() - ta
                ts = time.time()
                obs, reward, done, info = env.step(action)
                env_time += time.time() - ts
                steps += 1
                if done:
                    terminated = bool(info.get("terminated", False))
            rep_rows.append({
                "rank": rank,
                "repeat": rep,
                "episode": ep,
                "steps": steps,
                "completed_600": steps >= args.max_steps,
                "terminated": terminated,
                "environment_time_s": env_time,
                "policy_time_s": policy_time,
            })
        comm.Barrier()
        wall = time.time() - t0
        local_steps = sum(r["steps"] for r in rep_rows)
        local_env_time = sum(r["environment_time_s"] for r in rep_rows)
        local_policy_time = sum(r["policy_time_s"] for r in rep_rows)
        total_steps = comm.allreduce(local_steps, MPI.SUM)
        total_episodes = comm.allreduce(len(rep_rows), MPI.SUM)
        completed = comm.allreduce(
            sum(1 for r in rep_rows if r["completed_600"]), MPI.SUM)
        falls = comm.allreduce(
            sum(1 for r in rep_rows if r["terminated"]), MPI.SUM)
        max_env_time = comm.allreduce(local_env_time, MPI.MAX)
        max_policy_time = comm.allreduce(local_policy_time, MPI.MAX)
        repeats.append({
            "repeat": rep,
            "wall_time_s": float(wall),
            "aggregate_steps": int(total_steps),
            "episodes": int(total_episodes),
            "completed_600": int(completed),
            "terminated_falls": int(falls),
            "max_rank_environment_time_s": float(max_env_time),
            "max_rank_policy_time_s": float(max_policy_time),
            "env_steps_per_second": (float(total_steps) / max_env_time
                                     if max_env_time > 0 else None),
            "wall_steps_per_second": (float(total_steps) / wall
                                      if wall > 0 else None),
        })
        rows.extend(rep_rows)
        if rank == 0:
            print("[env-bench] repeat=%d steps=%d eps=%d completed=%d falls=%d "
                  "env_time=%.2fs wall=%.2fs env_steps/s=%.1f wall_steps/s=%.1f"
                  % (rep, total_steps, total_episodes, completed, falls,
                     max_env_time, wall,
                     repeats[-1]["env_steps_per_second"] or 0.0,
                     repeats[-1]["wall_steps_per_second"] or 0.0))

    if rank == 0:
        payload = {
            "run_id": args.run_id,
            "tag": args.tag,
            "motion_file": motion_file,
            "policy_file": policy_file,
            "policy_sha256": sha256_file(policy_file),
            "config": {
                "mpi_processes": num_procs,
                "repeats": args.repeats,
                "episodes_per_rank": args.episodes_per_rank,
                "max_steps": args.max_steps,
                "episode_mode": "test (fixed 600-step limit, no curriculum)",
                "randomizer": bool(args.enable_randomizer),
                "deterministic_policy": True,
                "timing_scope": "env.step wall time only; policy.predict timed separately",
            },
            "runtime": {
                "python": sys.version.split()[0],
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "finished_at_utc": utc_now(),
            },
            "repeats": repeats,
            "aggregate": {
                "total_steps": int(sum(r["aggregate_steps"] for r in repeats)),
                "total_episodes": int(sum(r["episodes"] for r in repeats)),
                "completed_600": int(sum(r["completed_600"] for r in repeats)),
                "terminated_falls": int(sum(r["terminated_falls"] for r in repeats)),
                "env_steps_per_second_mean": float(np.mean(
                    [r["env_steps_per_second"] for r in repeats])),
                "env_steps_per_second_min": float(np.min(
                    [r["env_steps_per_second"] for r in repeats])),
                "env_steps_per_second_max": float(np.max(
                    [r["env_steps_per_second"] for r in repeats])),
                "wall_steps_per_second_mean": float(np.mean(
                    [r["wall_steps_per_second"] for r in repeats])),
            },
            "rank0_rows": rows,
        }
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "benchmark.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        with open(os.path.join(out_dir, "episodes.csv"), "w",
                  newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["rank", "repeat", "episode", "steps",
                             "completed_600", "terminated",
                             "environment_time_s", "policy_time_s"])
            for row in rows:
                writer.writerow([row["rank"], row["repeat"], row["episode"],
                                 row["steps"], row["completed_600"],
                                 row["terminated"], row["environment_time_s"],
                                 row["policy_time_s"]])
        print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))

    env.close()


if __name__ == "__main__":
    main()
