import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_encoded(model, state, detach=True):
    z = model.encode(state.to(device))[0]
    if detach:
        return z.detach()
    else:
        return z

def get_encoded_raw(model, state, detach=True):
    z = model.encode(torch.from_numpy(state.copy()).to(device).float().unsqueeze(0).swapaxes(1, 3))[0]
    if detach:
        return z.detach()
    else:
        return z

def get_reshaped(state, scale=True):
    scale_factor = (1/255.0) if scale else 1.0
    return np.swapaxes(state.copy(), 0, 2)* scale_factor