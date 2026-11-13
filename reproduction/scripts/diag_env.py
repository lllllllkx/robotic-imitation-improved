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
print("env dt", env.env_time_step)
print("sim base", pyb.getBasePositionAndOrientation(sim)[0])
print("ref base", pyb.getBasePositionAndOrientation(ref)[0])
print("task ref pose root", task._ref_pose[:3])
print("task _get_ref_base_position", task._get_ref_base_position())
print("motion duration", task.get_active_motion().get_duration())
print("ref model numjoints", pyb.getNumJoints(ref), "sim", pyb.getNumJoints(sim))
print("feet", env.robot._foot_link_ids)
