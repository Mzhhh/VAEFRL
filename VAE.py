import os
import shutil
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
	def __init__(self, num_channel, a):
		super(UnFlatten, self).__init__()
		self.num_channel = num_channel
		self.a = a

	def forward(self, input):
		return input.view(input.size(0), self.num_channel, self.a, self.a)

class CNNVAE(nn.Module):
	def __init__(self, image_channels=3, h_dim=256, z_dim=32, unflatten_channel=256, unflatten_size=1):
		super(CNNVAE, self).__init__()
		self.encoder = nn.Sequential(
			nn.Conv2d(image_channels, 32, kernel_size=4, stride=3),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=3),  
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 128, kernel_size=4, stride=2),  
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, kernel_size=4, stride=2),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			Flatten()
		).to(device)
		
		self.fc1 = nn.Linear(h_dim, z_dim).to(device)
		self.fc2 = nn.Linear(h_dim, z_dim).to(device)
		self.fc3 = nn.Linear(z_dim, h_dim).to(device)
		
		self.decoder = nn.Sequential(
			UnFlatten(unflatten_channel, unflatten_size),
			nn.ConvTranspose2d(unflatten_channel, 128, kernel_size=4, stride=2),  
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=3),
			nn.BatchNorm2d(32),
			nn.ReLU(),
			nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=3),
			nn.BatchNorm2d(image_channels),
			nn.Sigmoid(),
		).to(device)
		
	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		# return torch.normal(mu, std)
		esp = torch.randn(*mu.size()).to(device)
		z = mu + std * esp
		return z
	
	def bottleneck(self, h):
		mu, logvar = self.fc1(h), self.fc2(h)
		z = self.reparameterize(mu, logvar)
		return z, mu, logvar

	def encode(self, x):
		h = self.encoder(x)
		z, mu, logvar = self.bottleneck(h)
		return z, mu, logvar

	def decode(self, z):
		z = self.fc3(z)
		z = self.decoder(z)
		return z

	def forward(self, x):
		z, mu, logvar = self.encode(x)
		z = self.decode(z)
		return z, mu, logvar


	def save(self, filename):
		torch.save(self.state_dict(), filename)

	def load(self, filename):
		self.load_state_dict(torch.load(filename))
		

def VAELoss(original, reconstructed, mu, logvar, KL_weight=1, writer_info=None):

	recon_loss = nn.L1Loss(reduction='mean')(original, reconstructed)
	KL_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

	if writer_info is not None:
		writer, t = writer_info
		writer.add_scalar("vae/recon_loss", recon_loss, t+1)
		writer.add_scalar("vae/kl_loss", KL_loss, t+1)

	return recon_loss + KL_loss * KL_weight


def generate_results(model, input):

	vae_input = torch.from_numpy(input.copy()).to(device).float().swapaxes(1, 3)
	latent = model.encode(vae_input)[0].detach()
	vae_recon = model(vae_input)[0].detach()

	latent_np = latent.cpu().numpy()
	vae_recon_np = vae_recon.cpu().numpy()

	return latent_np, vae_recon_np


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--num_sample", default=1e3, type=float)
	parser.add_argument("--show_image", default=-1, type=float)
	parser.add_argument("--max_episode_steps", default=1e2, type=float) 
	parser.add_argument("--load_model", default="", type=str)    
	args = parser.parse_args()

	num_sample = int(args.num_sample)
	show_image = int(args.show_image) if args.show_image > 0 else num_sample
	assert show_image <= num_sample, f"Cannot show more than {num_sample} samples"

	max_episode_steps = int(args.max_episode_steps)
	VAE_MODEL_FILE = args.load_model
	Z_DIM = 32

	# generate input
	state_array = np.zeros((num_sample, 96, 96, 3))
	collected = 0

	from pyvirtualdisplay import Display
	display = Display(visible=0, size=(400, 300))
	display.start()

	import gym
	env_name = "CarRacing-v0"
	env = gym.make(env_name)

	need_reset = True
	episode_step = 0
	while collected < num_sample:
		if need_reset:
			state, done = env.reset(), False
			episode_step = 0
		else:
			state, _, done, _ = env.step(env.action_space.sample())
			episode_step += 1
		state_array[collected, :] = state.copy()
		collected += 1
		need_reset = done or (episode_step >= max_episode_steps)
	
	vae = CNNVAE(image_channels=3, h_dim=256, z_dim=32)
	vae.load(os.path.join("./model_checkpoints", VAE_MODEL_FILE))
	vae.eval()

	latent_np, vae_recon_np = generate_results(vae, state_array/255.0)
	vae_recon_np = (vae_recon_np.swapaxes(1, 3) * 255).astype(np.uint8).copy()

	if os.path.exists("./tmp"):
		shutil.rmtree("./tmp")
	os.mkdir("./tmp")
	
	# plot latent distributions
	os.mkdir("./tmp/latent")
	from matplotlib import pyplot as plt
	import seaborn as sns

	for z in range(Z_DIM):
		_, ax = plt.subplots(figsize=(6, 6))
		sns.histplot(latent_np[:, z].flatten(), ax=ax)
		plt.savefig(f"./tmp/latent/dim{z+1}.png", dpi=200, bbox_inches="tight")
		plt.close()

	# plot latent distributions
	os.mkdir("./tmp/reconstruction")
	show_index = np.random.choice(num_sample, show_image, replace=False)
	for i, index in enumerate(show_index, start=1):
		orig = state_array[int(index)]
		recon = vae_recon_np[int(index)]
		_, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
		ax1.imshow(orig)
		ax2.imshow(recon)
		plt.savefig(f"./tmp/reconstruction/pair{i}.png", dpi=200, bbox_inches="tight")
		plt.close()


		