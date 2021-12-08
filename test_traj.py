import pickle
import os
import sys
import numpy as np
import gym
from utils.utils import generate_pairs, process_expert_traj, generate_tuples, adjust_lr
env = gym.make("Hopper-v2")
print(env.observation_space)
print(env.action_space)
path = "./expert/assets/expert_traj/Hopper-v2_expert_traj.p"
expert_traj_raw = pickle.load(open(path, "rb"))
if isinstance(expert_traj_raw, np.ndarray):
    expert_traj_raw_list = []
    for i in range(len(expert_traj_raw)):
        expert_traj_raw_list.append(expert_traj_raw[i])
    expert_traj_raw = expert_traj_raw_list

expert_traj = process_expert_traj(expert_traj_raw)
print("OK")