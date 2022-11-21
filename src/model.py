import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import random
from skimage import io
from collections import deque
import warnings
warnings.filterwarnings("ignore")


class QNetwork(nn.Module):
    """Q-network"""

    def __init__(self, n_inputs, n_outputs, learning_rate, weight_decay):
        super(QNetwork, self).__init__()
        n_hidden = 2500

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=8,
                               stride=4,
                               padding=0)
        #self.pool1 = nn.MaxPool2d(3,stride=2)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2,
                               padding=0)
        #self.pool2 = nn.MaxPool2d(3,stride=2)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=0)

        #self.linear = nn.Linear(960, n_hidden, bias=True)
        #torch.nn.init.normal_(self.linear.weight, 0, 1)

        self.flat_dim = n_inputs
        self.out = nn.Linear(self.flat_dim, n_outputs, bias=True)
        torch.nn.init.normal_(self.out.weight, 0, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        #x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        #x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        b_size = x.shape[0]
        x = x.reshape((b_size, self.flat_dim))
        x = self.out(x)
        return x

    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))

    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)
    
    def save(self, episode_num, epsilon, model_path):
        torch.save({'episode_num': episode_num,
                    'epsilon': epsilon,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'model_state_dict': self.state_dict()},
                    model_path,
                    _use_new_zipfile_serialization=False)


class ReplayMemory(object):
    """Experience Replay Memory"""

    def __init__(self, capacity):
        #self.size = size
        self.memory = deque(maxlen=capacity)

    def add(self, s, a, r, s1, d):
        """Add experience to memory."""
        s *= 255
        s = s.astype('uint8')

        s1 *= 255
        s1 = s1.astype('uint8')
        self.memory.append([s, a, r, s1, d])

    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.sample(self.memory, batch_size)

    def count(self):
        return len(self.memory)
