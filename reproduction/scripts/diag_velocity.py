"""Check how the reference model's joint velocities behave (reward term sanity)."""
import os, sys
code = r"C:\Users\88950\Doubao\chats\2026-11-13\new-chat-1\robotic_imitation_repro\code\motion_imitation"
sys.path.insert(0, code)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.chdir(code)
import numpy as np
import envs.env_builder as env_builder

env = env_builder.build_imitation_env(
    motion_files=["data/motions/dog_pace.txt"], num_parallel_envs=1,
    mode="test", enable_randomizer=False, enable_rendering=False)
task = env._task
task._enable_rand_init_time = False
task._ref_state_init_prob = 1.0
o = env.reset()
pyb = env._pybullet_client
sim = env.robot.quadruped
ref = task._ref_model
for i in range(6):
  a = env.action_space.sample()
  o, r, done, info = env.step(a)
  ref_v = [float(pyb.getJointStateMultiDof(ref, j)[1][0]) if len(pyb.getJointStateMultiDof(ref, j)[1]) else 0.0 for j in range(pyb.getNumJoints(ref))]
  sim_v = [float(pyb.getJointStateMultiDof(sim, j)[1][0]) if len(pyb.getJointStateMultiDof(sim, j)[1]) else 0.0 for j in range(pyb.getNumJoints(sim))]
  ref_q = [float(pyb.getJointStateMultiDof(ref, j)[0][0]) if len(pyb.getJointStateMultiDof(ref, j)[0]) else 0.0 for j in range(pyb.getNumJoints(ref))]
  print("step", i, "reward %.3f" % r)
  print("  ref_q  ", np.round(ref_q, 3))
  print("  ref_dq ", np.round(ref_v, 4))
  print("  sim_dq ", np.round(sim_v, 4))
  print("  vel_reward %.4f pose_reward %.4f" % (task._calc_reward_velocity(), task._calc_reward_pose()))
