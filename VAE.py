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
    def __init__(self, image_channels=3, h_dim=4096, z_dim=32, unflatten_channel=256, unflatten_size=4):
        super(CNNVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),  # b, 32, 47, 47
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # b, 64, 22, 22
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),  # b, 128, 10, 10
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),  # b, 256, 4, 4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Flatten()  # b, 4096
        ).to(device)
        
        self.fc1 = nn.Linear(h_dim, z_dim).to(device)
        self.fc2 = nn.Linear(h_dim, z_dim).to(device)
        self.fc3 = nn.Linear(z_dim, h_dim).to(device)
        
        self.decoder = nn.Sequential(
            UnFlatten(unflatten_channel, unflatten_size),  # b, 4096, 1, 1
            nn.ConvTranspose2d(unflatten_channel, 128, kernel_size=4, stride=2),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
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