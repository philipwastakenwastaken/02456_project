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




class QNetwork(nn.Module):
    """Q-network"""

    def __init__(self, n_inputs, n_outputs, learning_rate):
        super(QNetwork, self).__init__()
        n_hidden = 2500
        
        self.conv1 = nn.Conv2d(in_channels=1,
                             out_channels=32,
                             kernel_size=8,
                             stride=4,
                             padding=0)
        self.pool1 = nn.MaxPool2d(3,stride=2)
        
<<<<<<< HEAD
        self.conv2 = nn.Conv2d(in_channels=32,
                             out_channels=64,
                             kernel_size=4,
                             stride=2,
                             padding=0)
        self.pool2 = nn.MaxPool2d(3,stride=2)
        
        # self.conv3 = nn.Conv2d(in_channels=64,
        #                      out_channels=128,
        #                      kernel_size=2,
        #                      stride=1,
        #                      padding=0)
        # self.pool3 = nn.MaxPool2d(3,stride=2)
        
=======
>>>>>>> 1b950d5ef6d8c4813022d3458b1c290de1f7ee36
        #self.linear = nn.Linear(960, n_hidden, bias=True)
        #torch.nn.init.normal_(self.linear.weight, 0, 1)

        self.out = nn.Linear(15200, n_outputs, bias=True)
        torch.nn.init.normal_(self.out.weight, 0, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    
    def forward(self, x):
        x = x / 255.0
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        b_size = x.shape[0]
        x = x.reshape((b_size,15200))
        x = self.out(x)
        return x
    

    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))


    def update_params(self, new_params, tau):
        params = self.state_dict()
        for k in params.keys():
            params[k] = (1-tau) * params[k] + tau * new_params[k]
        self.load_state_dict(params)



class ReplayMemory(object):
    """Experience Replay Memory"""
    
    def __init__(self, capacity):
        #self.size = size
        self.memory = deque(maxlen=capacity)
    
    def add(self, *args):
        """Add experience to memory."""
        self.memory.append([*args])
    
    def sample(self, batch_size):
        """Sample batch of experiences from memory with replacement."""
        return random.sample(self.memory, batch_size)
    
    def count(self):
        return len(self.memory)

def get_parameters():
    input_size = 22528
    output_size = 9
    learning_rate = 3e-4 
    return input_size, output_size, learning_rate