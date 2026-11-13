"""Evaluate a pretrained motion-imitation policy and dump per-step imitation data.

This script does NOT modify the upstream repository. It imports the upstream
package from --code_dir and drives the same environment builder / PPO model
that motion_imitation/run.py uses, while logging the quantities the paper
reports (reward and its terms, joint tracking error, root tracking error).
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--code_dir", required=True,
                      help="Path to the cloned motion_imitation package root "
                           "(the directory that contains 'envs', 'learning', ...)")
  parser.add_argument("--motion_file", required=True)
  parser.add_argument("--model_file", required=True)
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--tag", default=None)
  parser.add_argument("--episodes", type=int, default=3)
  parser.add_argument("--video", action="store_true")
  parser.add_argument("--video_every", type=int, default=1)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--max_steps", type=int, default=1000)
  return parser.parse_args()


def main():
  args = parse_args()
  code_dir = os.path.abspath(args.code_dir)
  sys.path.insert(0, code_dir)
  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

  import tensorflow as tf
  import envs.env_builder as env_builder
  import learning.imitation_policies as imitation_policies
  import learning.ppo_imitation as ppo_imitation

  tag = args.tag or os.path.splitext(os.path.basename(args.motion_file))[0]
  out_dir = os.path.join(args.out_dir, tag)
  os.makedirs(out_dir, exist_ok=True)
  frame_dir = os.path.join(out_dir, "frames")
  if args.video:
    os.makedirs(frame_dir, exist_ok=True)

  print("[eval] building env for %s" % args.motion_file)
  env = env_builder.build_imitation_env(
      motion_files=[args.motion_file],
      num_parallel_envs=1,
      mode="test",
      enable_randomizer=False,
      enable_rendering=False)

  # Deterministic, paper-style evaluation: every episode starts at t=0 from
  # the reference state (no random time offset), so episodes are comparable.
  task = env._task
  task._enable_rand_init_time = False
  task._ref_state_init_prob = 1.0

  policy_kwargs = {
      "net_arch": [{"pi": [512, 256], "vf": [512, 256]}],
      "act_fun": tf.nn.relu,
  }
  model = ppo_imitation.PPOImitation(
      policy=imitation_policies.ImitationPolicy,
      env=env,
      gamma=0.95,
      timesteps_per_actorbatch=4096,
      clip_param=0.2,
      optim_epochs=1,
      optim_stepsize=1e-5,
      optim_batchsize=256,
      lam=0.95,
      adam_epsilon=1e-5,
      schedule="constant",
      policy_kwargs=policy_kwargs,
      verbose=0)

  print("[eval] loading policy %s" % args.model_file)
  model.load_parameters(args.model_file)

  pyb = env._pybullet_client
  robot = env.robot
  sim_model = robot.quadruped
  ref_model = task._ref_model
  n_joints = pyb.getNumJoints(ref_model)
  foot_ids = list(robot._foot_link_ids)
  env_dt = env.env_time_step
  num_motors = len([j for j in range(n_joints)
                    if len(pyb.getJointStateMultiDof(sim_model, j)[0]) > 0])

  # Camera follows the simulated robot from a fixed relative vantage point.
  def render_frame():
    base_pos, _ = pyb.getBasePositionAndOrientation(sim_model)
    cam_target = [base_pos[0], base_pos[1], 0.35]
    view = pyb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=cam_target, distance=1.7, yaw=52, pitch=-24,
        roll=0, upAxisIndex=2)
    proj = pyb.computeProjectionMatrixFOV(
        fov=60, aspect=float(args.width) / args.height, nearVal=0.1,
        farVal=100)
    _, _, px, _, _ = pyb.getCameraImage(
        width=args.width, height=args.height, viewMatrix=view,
        projectionMatrix=proj, renderer=pyb.ER_TINY_RENDERER)
    rgb = np.asarray(px, dtype=np.uint8)[:, :, :3]
    return rgb

  csv_path = os.path.join(out_dir, "steps.csv")
  header = (["episode", "step", "time", "reward", "pose_reward",
             "velocity_reward", "end_effector_reward", "root_pose_reward",
             "root_velocity_reward", "pose_err_rad", "root_pos_err_m",
             "root_err_xy_m", "sim_z", "ref_z", "speed_mps", "yaw_err_rad"]
            + ["q%d" % i for i in range(num_motors)]
            + ["qref%d" % i for i in range(num_motors)]
            + ["foot_z%d" % i for i in range(len(foot_ids))])
  fh = open(csv_path, "w", newline="")
  writer = csv.writer(fh)
  writer.writerow(header)

  summaries = []
  for ep in range(args.episodes):
    o = env.reset()
    ep_reward = 0.0
    rows = []
    steps = 0
    t_ep = time.time()
    prev_pos = np.array(pyb.getBasePositionAndOrientation(sim_model)[0])

    while steps < args.max_steps:
      a, _ = model.predict(o, deterministic=True)
      o, r, done, info = env.step(a)
      ep_reward += r
      steps += 1

      ref_pose = task._ref_pose
      motion = task.get_active_motion()
      ref_joints = np.array(motion.get_frame_joints(ref_pose))

      sim_joints = []
      for j in range(n_joints):
        jq = pyb.getJointStateMultiDof(sim_model, j)[0]
        if len(jq) > 0:
          sim_joints.append(float(jq[0]))
      sim_joints = np.array(sim_joints)

      sim_pos = np.array(pyb.getBasePositionAndOrientation(sim_model)[0])
      ref_pos = np.array(pyb.getBasePositionAndOrientation(ref_model)[0])
      sim_orn = np.array(pyb.getBasePositionAndOrientation(sim_model)[1])

      pose_err = float(np.mean(np.abs(ref_joints - sim_joints)))
      root_err = float(np.linalg.norm(sim_pos - ref_pos))
      root_err_xy = float(np.linalg.norm((sim_pos - ref_pos)[:2]))
      speed = float(np.linalg.norm(sim_pos - prev_pos) / env_dt)
      prev_pos = sim_pos

      # heading error about z
      yaw_sim = 2.0 * np.arctan2(sim_orn[2], sim_orn[3])
      ref_orn = np.array(pyb.getBasePositionAndOrientation(ref_model)[1])
      yaw_ref = 2.0 * np.arctan2(ref_orn[2], ref_orn[3])
      yaw_err = float(abs((yaw_sim - yaw_ref + np.pi) % (2 * np.pi) - np.pi))

      terms = [task._calc_reward_pose(), task._calc_reward_velocity(),
               task._calc_reward_end_effector(), task._calc_reward_root_pose(),
               task._calc_reward_root_velocity()]

      foot_z = [float(pyb.getLinkState(sim_model, fid)[0][2]) for fid in foot_ids]

      row = ([ep, steps, steps * env_dt, r] + terms
             + [pose_err, root_err, root_err_xy,
                float(sim_pos[2]), float(ref_pos[2]), speed, yaw_err]
             + list(sim_joints) + list(ref_joints) + foot_z)
      rows.append(row)
      writer.writerow(row)

      if args.video and (steps % args.video_every == 0):
        from PIL import Image
        Image.fromarray(render_frame()).save(
            os.path.join(frame_dir, "ep%02d_%04d.png" % (ep, steps)))

      if done:
        break

    duration = time.time() - t_ep
    arr = np.array([row[3:] for row in rows], dtype=float)
    summary = {
        "tag": tag,
        "episode": ep,
        "steps": steps,
        "episode_seconds": steps * env_dt,
        "sum_reward": float(np.sum(arr[:, 0])),
        "mean_reward": float(np.mean(arr[:, 0])),
        "mean_pose_err_rad": float(np.mean(arr[:, 6])),
        "mean_root_err_m": float(np.mean(arr[:, 7])),
        "max_root_err_m": float(np.max(arr[:, 7])),
        "mean_speed_mps": float(np.mean(arr[:, 11])),
        "mean_sim_z": float(np.mean(arr[:, 9])),
        "mean_term_pose_reward": float(np.mean(arr[:, 1])),
        "mean_term_velocity_reward": float(np.mean(arr[:, 2])),
        "mean_term_end_effector_reward": float(np.mean(arr[:, 3])),
        "mean_term_root_pose_reward": float(np.mean(arr[:, 4])),
        "mean_term_root_velocity_reward": float(np.mean(arr[:, 5])),
        "wall_time_s": duration,
        "steps_per_sec": steps / max(duration, 1e-9),
        "terminated_early": bool(steps < args.max_steps),
    }
    summaries.append(summary)
    print("[eval] ep %d: steps=%d sum_reward=%.2f mean_reward=%.4f "
          "pose_err=%.4f rad root_err=%.3f m" %
          (ep, steps, summary["sum_reward"], summary["mean_reward"],
           summary["mean_pose_err_rad"], summary["mean_root_err_m"]))
  fh.close()

  with open(os.path.join(out_dir, "summary.json"), "w") as f:
    json.dump({"tag": tag, "motion_file": args.motion_file,
               "model_file": args.model_file,
               "env_time_step": env_dt,
               "num_motors": num_motors,
               "episodes": summaries}, f, indent=2)

  # ---- video assembly -------------------------------------------------------
  if args.video:
    frames = sorted(f for f in os.listdir(frame_dir) if f.startswith("ep00_"))
    if frames:
      from PIL import Image
      imgs = [np.asarray(Image.open(os.path.join(frame_dir, f))) for f in frames]
      fps = 30
      video_path = os.path.join(os.path.dirname(out_dir), tag + ".mp4")
      try:
        import imageio
        imageio.mimsave(video_path, imgs, fps=fps, quality=8)
        print("[eval] video -> %s" % video_path)
      except Exception as e:  # fallback to gif
        print("[eval] mp4 failed (%s), writing gif" % e)
        video_path = os.path.join(os.path.dirname(out_dir), tag + ".gif")
        pil = [Image.fromarray(f) for f in imgs]
        pil[0].save(video_path, save_all=True, append_images=pil[1:],
                    duration=int(1000 / fps), loop=0, optimize=True)
        print("[eval] video -> %s" % video_path)

      # filmstrip of key frames for the report
      picks = np.linspace(0, len(imgs) - 1, 6).astype(int)
      strip = np.concatenate([imgs[i] for i in picks], axis=1)
      Image.fromarray(strip).save(os.path.join(os.path.dirname(out_dir),
                                               tag + "_filmstrip.png"))
      print("[eval] filmstrip -> %s" % tag + "_filmstrip.png")

  env.close()
  print("[eval] done: %s" % out_dir)


if __name__ == "__main__":
  main()
