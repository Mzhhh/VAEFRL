import os

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

from VAE import *
from util import *

from DQN.CarRacingDQNAgent import CarRacingDQNAgent
from DQN.common_functions import *

from TD3 import DDPG, DDPG_CNN
from TD3.utils import ReplayBuffer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --- Hyperparameters START --- ###

VAE_MODEL_FILE = ""

REPLAY_BUFFER_SIZE = int(1e4)

max_timesteps = int(3e4)
expl_noise = 0.01
max_episode_steps = 100
batch_size = 128

lr = 1e-4

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

policy_repr = DDPG.DDPG(32, action_dim, min_action, max_action)
buffer_repr = ReplayBuffer([32], action_dim, REPLAY_BUFFER_SIZE, device=device)

vae = CNNVAE(image_channels=3, h_dim=4096, z_dim=32)
vae.load(VAE_MODEL_FILE)

log_writer = SummaryWriter(log_dir="./tensorboard/"+time.strftime("%m%d%H%M", time.localtime()), comment="logWriter")

### --- TRAINING START --- ### 


# step 3: fix VAE, train agent

state, done = env.reset(), False
state_repr = get_encoded_raw(vae, state).cpu().numpy()


for t in tqdm(range(max_timesteps)):
		
    episode_timesteps += 1

    # Select action according to expert model
    
    action = policy_repr.select_action(state_repr)

    action = (action + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(min_action, max_action)

    # Perform action
    next_state, reward, done, _ = env.step(action)
    next_state_repr = get_encoded_raw(vae, next_state).cpu().numpy()

    done_bool = float(done or (episode_timesteps > max_episode_steps))

    # Store data in replay buffer

    state_repr = state_repr.copy()
    next_state_repr = next_state_repr.copy()

    buffer_repr.add_vector(state_repr, action, next_state_repr, reward, done_bool)

    state_repr = next_state_repr

    episode_reward += reward

    # Train agent after collecting sufficient data
    if t >= start_timesteps:
        
        ### TRAINING ROUTINE START ###

        cr_loss, ac_loss = policy_repr.train(buffer_repr)
        log_writer.add_scalar("critic/loss", cr_loss, t+1)
        log_writer.add_scalar("agent/loss", ac_loss, t+1)

        ### TRAINING ROUTINE END   ###

    if done_bool: 

        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        log_writer.add_scalar("agent/reward", episode_reward, t+1)
        # Reset environment
        state, done = env.reset(), False
        state_repr = get_encoded_raw(vae, state).cpu().numpy()
        
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1 

        if episode_num % 50 == 0:
            if not os.path.exists("./model_checkpoints"):
                os.makedirs("./model_checkpoints")
            time_str = time.strftime("%m%d%H%M", time.localtime())
            vae.save("./model_checkpoints/agent_eps_%d_%s" % (episode_num, time_str))


### --- TRAINING END --- ###

