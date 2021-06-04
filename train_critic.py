import os
import argparse

import numpy as np
from tqdm import tqdm
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import gym
import time

# stolen form others' repos

from DQN.CarRacingDQNAgent import CarRacingDQNAgent
from DQN.common_functions import *

from TD3 import DDPG, DDPG_CNN
from TD3.utils import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --- Hyperparameters START --- ###

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--max_timesteps", default=1e6, type=int)    # Max time steps to run environment
parser.add_argument("--buffer_size", default=1e6, type=int)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=128, type=int)       # Batch size for both actor and critic
args = parser.parse_args()

REPLAY_BUFFER_SIZE = int(1e4)

max_timesteps = int(args.max_timesteps)
expl_noise = args.expl_noise
max_episode_steps = args.max_timesteps
batch_size = args.batch_size

eval_freq = -1  # expert model
start_timesteps = 0  # expert model

### --- Hyperparameters END   --- ###


env_name = "CarRacing-v0"
env = gym.make(env_name)

episode_reward = 0
episode_timesteps = 0
episode_num = 0

state_dim_image = env.observation_space.shape
action_dim = env.action_space.shape[0]

min_action = env.action_space.low
max_action = env.action_space.high


# model components

buffer_raw = ReplayBuffer((3, 96, 96), action_dim, REPLAY_BUFFER_SIZE, device=device)
policy_raw = DDPG_CNN.DDPG(3, action_dim, min_action, max_action)

expert_model = CarRacingDQNAgent(epsilon=0)
expert_model.load("tf_best.h5")

log_writer = SummaryWriter(log_dir="./tensorboard/"+time.strftime("%m%d%H%M", time.localtime()), comment="logWriter")


### --- TRAINING START --- ### 

state, done = env.reset(), False

state_tf = process_state_image(state)  # for tf model

state_frame_stack_queue = deque([state_tf]*3, maxlen=3)

for t in tqdm(range(max_timesteps)):
		
    episode_timesteps += 1

    # Select action according to expert model
    
    current_state_frame_stack = generate_state_frame_stack_from_queue(state_frame_stack_queue)
    action = expert_model.act(current_state_frame_stack)
    action = (action + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(min_action, max_action)

    # Perform action
    next_state, reward, done, _ = env.step(action)

    next_state_tf = process_state_image(next_state)  # for tf
    state_frame_stack_queue.append(next_state_tf)

    done_bool = float(done or (episode_timesteps > max_episode_steps))

    # Store data in replay buffer
    state = state.copy()
    next_state = next_state.copy()
    buffer_raw.add(state, action, next_state, reward, done_bool)

    state = next_state.copy()
    state_tf = next_state_tf

    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= start_timesteps:
        
        ### TRAINING ROUTINE START ###

        batch = buffer_raw.sample(batch_size)  # (state, action, next_state, reward, not_done)

        # update policy_raw

        policy_raw.clear_gradient()

        critic_loss_raw = policy_raw.critic_loss(batch)
        critic_loss_raw.backward()
        policy_raw.critic_optimizer.step()

        policy_raw.update_weights()

        log_writer.add_scalar("policy_raw/critic_loss", critic_loss_raw.data, t+1)

        ### TRAINING ROUTINE END   ###

    if done_bool: 

        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        log_writer.add_scalar("agent/reward", episode_reward, t+1)
        
        # Reset environment
        # clear_output(wait=True)
        state, done = env.reset(), False

        state_tf = process_state_image(state)  # for tf model
        state_frame_stack_queue = deque([state_tf]*3, maxlen=3)

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 

        if episode_num % 50 == 0:
            if not os.path.exists("./model_checkpoints"):
                os.makedirs("./model_checkpoints")
            time_str = time.strftime("%m%d%H%M", time.localtime())
            policy_raw.save("./model_checkpoints/critic_eps_%d_%s" % (episode_num, time_str))

### --- TRAINING END --- ###