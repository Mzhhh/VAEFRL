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

from VAE import *
from util import *

from DQN.CarRacingDQNAgent import CarRacingDQNAgent
from DQN.common_functions import *

from TD3 import DDPG, DDPG_CNN
from TD3.utils import ReplayBuffer



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --- Hyperparameters START --- ###

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=float) # Time steps initial random policy is used
parser.add_argument("--max_timesteps", default=1e6, type=float)    # Max time steps to run environment
parser.add_argument("--max_episode_steps", default=1e2, type=float)
parser.add_argument("--buffer_size", default=1e6, type=float)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=128, type=float)       # Batch size for both actor and critic
parser.add_argument("--learning_rate", default=1e-4)                      # Target network update rate
parser.add_argument("--load_model", default="", type=str)                  # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--eval_freq", default=50, type=float)
parser.add_argument("--virtual_display", action="store_true")
args = parser.parse_args()


VAE_MODEL_FILE = args.load_model

REPLAY_BUFFER_SIZE = int(args.buffer_size)

max_timesteps = int(args.max_timesteps)
expl_noise = args.expl_noise
max_episode_steps = int(args.max_episode_steps)
batch_size = int(args.batch_size)
eval_freq = int(args.eval_freq)

lr = args.learning_rate

eval_freq = -1
start_timesteps = args.start_timesteps

### --- Hyperparameters END   --- ###


# setup virtual display

if args.virtual_display:

	from pyvirtualdisplay import Display
	from IPython.display import clear_output
	display = Display(visible=0, size=(400, 300))
	display.start()


env_name = "CarRacing-v0"
env = gym.make(env_name)
env.seed(args.seed)

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

vae = CNNVAE(image_channels=3, h_dim=256, z_dim=32)
vae.load(os.path.join("./model_checkpoints", VAE_MODEL_FILE))

log_writer = SummaryWriter(log_dir="./tensorboard/"+time.strftime("%m%d%H%M", time.localtime()), comment="logWriter")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environments
def eval_policy(vae, policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		state = state / 255.0  # normalize
		state_repr = get_encoded_raw(vae, state).cpu().numpy()
		
		while not done:
			action = policy.select_action(state_repr)
			state, reward, done, _ = eval_env.step(action)
			state = state / 255.0
			state_repr = get_encoded_raw(vae, state).cpu().numpy()

			avg_reward += reward

	avg_reward /= eval_episodes

	return avg_reward


### --- TRAINING START --- ### 


# step 3: fix VAE, train agent

state, done = env.reset(), False
state = state / 255.0

state_repr = get_encoded_raw(vae, state).cpu().numpy()


for t in tqdm(range(max_timesteps)):
		
	episode_timesteps += 1

	# Select action according to expert model
	
	action = policy_repr.select_action(state_repr)

	action = (action + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(min_action, max_action)

	# Perform action
	next_state, reward, done, _ = env.step(action)
	next_state = next_state / 255.0

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
		log_writer.add_scalar("policy/episode_reward", episode_reward, t+1)
		# Reset environment
		state, done = env.reset(), False
		state = state / 255.0

		state_repr = get_encoded_raw(vae, state).cpu().numpy()
		
		episode_reward = 0
		episode_timesteps = 0
		episode_num += 1 

		if episode_num % 50 == 0:
			if not os.path.exists("./model_checkpoints"):
				os.makedirs("./model_checkpoints")
			time_str = time.strftime("%m%d%H%M", time.localtime())
			policy_repr.save("./model_checkpoints/agent_eps_%d_%s" % (episode_num, time_str))

		if episode_num % eval_freq == 0:
			avg_reward = eval_policy(vae, policy_repr, env_name, args.seed)
			log_writer.add_scaler("policy/eval_reward", avg_reward, t+1)



### --- TRAINING END --- ###

