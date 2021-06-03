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

from DQN.CarRacingDQNAgent import CarRacingDQNAgent
from DQN.common_functions import *

from TD3 import DDPG, DDPG_CNN
from TD3.utils import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --- Hyperparameters START --- ###

CRITIC_MODEL_FILE = ""

REPLAY_BUFFER_SIZE = int(1e4)

max_timesteps = int(3e4)
expl_noise = 0.01
max_episode_steps = 100
batch_size = 128

kl_weight = 0.1
consistency_weight = 1

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

buffer_raw = ReplayBuffer((3, 96, 96), action_dim, REPLAY_BUFFER_SIZE, device=device)
policy_raw = DDPG_CNN.load(os.path.join("./model_checkpoints", CRITIC_MODEL_FILE))

expert_model = CarRacingDQNAgent(epsilon=0)
expert_model.load("tf_best.h5")

vae = CNNVAE(image_channels=3, h_dim=4096, z_dim=32)
vae_optimizer = optim.Adam(vae.parameters(), lr=lr)

vae_loss = lambda original, reconstructed, mu, logvar, t: \
           VAELoss(original, reconstructed, mu, logvar, KL_weight=kl_weight, writer_info=(log_writer, t))

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

    state = next_state
    state_tf = next_state_tf

    episode_reward += reward

    # Train agent after collecting sufficient data

    if t >= start_timesteps:
        
        ### TRAINING ROUTINE START ###

        # update vae

        vae_optimizer.zero_grad()
        
        batch = buffer_raw.sample(batch_size)  # (state, action, next_state, reward, not_done)
        vae_input = batch[0]

        reconstruction, mu, logvar = vae(vae_input)  # get reconstructed images

        vae_loss_naked = vae_loss(vae_input, reconstruction, mu, logvar, t)

        Q_input = policy_raw.Q_value(batch).detach()
        Q_recon = policy_raw.Q_value((reconstruction, batch[1], None, None, None))  # only feed in reconstructed state & action
        Q_consistency_loss_1 = nn.MSELoss(reduction='sum')(Q_input, Q_recon)  # between original & reconstructed

        vae_loss_total = vae_loss_naked + consistency_weight * Q_consistency_loss_1

        vae_loss_total.backward()
        
        vae_optimizer.step()

        log_writer.add_scalar("vae/loss_naked", vae_loss_naked.data, t+1)
        log_writer.add_scalar("vae/consistency_loss", Q_consistency_loss_1.data, t+1)
        log_writer.add_scalar("vae/loss_total", vae_loss_total.data, t+1)

        ### TRAINING ROUTINE END   ###

    if done_bool: 

        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        log_writer.add_scalar("agent/reward", episode_reward, t+1)
        # Reset environment
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
            vae.save("./model_checkpoints/vae_eps_%d_%s" % (episode_num, time_str))

### --- TRAINING END --- ###

