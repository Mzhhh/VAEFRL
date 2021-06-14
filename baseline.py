import os
import argparse
import re

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

from TD3 import DDPG, DDPG_CNN
from TD3.utils import ReplayBuffer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### --- Hyperparameters START --- ###

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int)               # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=float) # Time steps initial random policy is used
parser.add_argument("--max_timesteps", default=1e6, type=float)    # Max time steps to run environment
parser.add_argument("--episode_start_steps", default=50, type=float)
parser.add_argument("--max_episode_steps", default=250, type=float)
parser.add_argument("--buffer_size", default=1e6, type=float)    # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                 # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=128, type=float)       # Batch size for both actor and critic
parser.add_argument("--learning_rate", default=1e-4)                      # Target network update rate
parser.add_argument("--eval_freq", default=50, type=float)
parser.add_argument("--virtual_display", action="store_true")
parser.add_argument("--constraint_action", action="store_true")
parser.add_argument("--pretrained_model", default="", type=str)    # load pretrained model 
parser.add_argument("--min_gas", default=0.6, type=float)
parser.add_argument("--max_gas", default=1.0, type=float)
parser.add_argument("--max_break", default=0.2, type=float)
parser.add_argument("--tag", default="", type=str)
args = parser.parse_args()

REPLAY_BUFFER_SIZE = int(args.buffer_size)

max_timesteps = int(args.max_timesteps)
expl_noise = args.expl_noise
episode_start_steps = int(args.episode_start_steps)
max_episode_steps = int(args.max_episode_steps)
batch_size = int(args.batch_size)
eval_freq = int(args.eval_freq)

lr = args.learning_rate

start_timesteps = args.start_timesteps

### --- Hyperparameters END   --- ###

# load from model checkpoint
PRETRAINED_MODEL = args.pretrained_model
if PRETRAINED_MODEL.lower() == "newest":
	avail_models = [f for f in os.listdir("./model_checkpoints") if f.startswith("agent")]
	avail_model_prefix = sorted(list(set([re.search(r"agent\_eps\_\d+\_\d+", f).group() for f in avail_models])), key=lambda s: s.split("_")[-1])
	assert avail_models, "No available critic model"
	PRETRAINED_MODEL = avail_model_prefix[0]
	print(f"Using latest version: {PRETRAINED_MODEL}")



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

if args.constraint_action:
	min_action = np.array([-1.0, args.min_gas, 0.0]).astype(np.float32)
	max_action = np.array([1.0, args.max_gas, args.max_break]).astype(np.float32)
else:
	min_action = env.action_space.low
	max_action = env.action_space.high

DEFAULT_ACTION = np.zeros(3).astype(np.float32)

# model components

buffer_raw = ReplayBuffer((3, 64, 64), action_dim, REPLAY_BUFFER_SIZE, device=device)
policy_raw = DDPG_CNN.DDPG(3, action_dim, min_action, max_action)
if PRETRAINED_MODEL:
	policy_raw.load(os.path.join("./model_checkpoints", PRETRAINED_MODEL))

TAG = args.tag
log_writer = SummaryWriter(log_dir="./tensorboard/"+time.strftime("%m%d%H%M", time.localtime())+TAG, comment="logWriter")
LOG_INTERVAL = 10


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environments
def eval_policy(policy, env_name, seed, eval_episodes=10, episode_start=episode_start_steps, episode_end=max_episode_steps):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		episode_step = 0
		state, done = eval_env.reset(), False
		state = clip_image(state)
		state = np.swapaxes(state, 0, 2)[np.newaxis, :].copy()
		
		while not done and episode_step < episode_end:
			action = DEFAULT_ACTION if episode_step < episode_start else policy.select_action(state)
			state, reward, done, _ = eval_env.step(action)
			state = clip_image(state)
			state = np.swapaxes(state, 0, 2)[np.newaxis, :].copy()
			episode_step += 1

			avg_reward += reward

	avg_reward /= eval_episodes

	return avg_reward


### --- TRAINING START --- ### 

state, done = env.reset(), False
for _ in range(episode_start_steps):
	state, _, _, _ = env.step(DEFAULT_ACTION)
state = clip_image(state)
state = np.swapaxes(state, 0, 2)[np.newaxis, :].copy()

for t in tqdm(range(max_timesteps)):
		
	episode_timesteps += 1

	# Select an action
	if t < start_timesteps:
		action = min_action + (max_action - min_action) * np.random.rand(*max_action.shape).astype(np.float32)
	else:
		action = policy_raw.select_action(state)
		action = (action + np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(min_action, max_action)

	# Perform action
	next_state, reward, done, _ = env.step(action)
	next_state = clip_image(next_state)
	next_state = np.swapaxes(next_state, 0, 2)[np.newaxis, :].copy()

	done_bool = float(done or (episode_timesteps > max_episode_steps))

	# Store data in replay buffer

	buffer_raw.add_vector(state, action, next_state, reward, done_bool)

	state = next_state

	episode_reward += reward

	# Train agent after collecting sufficient data
	if t >= start_timesteps:
		
		### TRAINING ROUTINE START ###

		cr_loss, ac_loss = policy_raw.train(buffer_raw)

		if t % LOG_INTERVAL == 1:
			log_writer.add_scalar("critic/loss", cr_loss, t+1)
			log_writer.add_scalar("actor/loss", ac_loss, t+1)

		### TRAINING ROUTINE END   ###

	if done_bool: 

		# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
		print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
		log_writer.add_scalar("agent/episode_reward", episode_reward, t+1)
		# Reset environment
		state, done = env.reset(), False
		for _ in range(episode_start_steps):
			state, _, _, _ = env.step(DEFAULT_ACTION)
		state = clip_image(state)
		state = np.swapaxes(state, 0, 2)[np.newaxis, :].copy()
		
		episode_reward = 0
		episode_timesteps = 0
		episode_num += 1 

		if episode_num % 100 == 0:
			if not os.path.exists("./model_checkpoints"):
				os.makedirs("./model_checkpoints")
			time_str = time.strftime("%m%d%H%M", time.localtime())
			policy_raw.save("./model_checkpoints/agent_eps_%d_%s" % (episode_num, time_str) + ("_%s"%TAG if TAG else ""))

		if t > start_timesteps and episode_num % eval_freq == 0:
			avg_reward = eval_policy(policy_raw, env_name, args.seed+1234, 5)
			log_writer.add_scalar("agent/eval_reward", avg_reward, t+1)

### --- TRAINING END --- ###

