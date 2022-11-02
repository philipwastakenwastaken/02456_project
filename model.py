import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from skimage import io


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
        
        #self.conv2 = nn.Conv2d(in_channels=32,
        #                     out_channels=64,
        #                     kernel_size=4,
        #                     stride=2,
        #                     padding=0)
        #self.pool2 = nn.MaxPool2d(3,stride=2)
        
        #self.conv3 = nn.Conv2d(in_channels=64,
        #                     out_channels=64,
        #                     kernel_size=3,
        #                     stride=1,
        #                     padding=0)
        #self.pool3 = nn.MaxPool2d(3,stride=2)
        
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

        x = x.flatten()
        x = self.out(x)
        
        #x = self.conv2(x)
        #x = F.relu(x)
        #x = self.pool2(x)
        
        #x = self.conv3(x)
        #x = self.pool3(x)
        #x = F.relu(x)

        #x = self.linear(x)
        #x = F.elu(x)

        return x
    
    def loss(self, q_outputs, q_targets):
        return torch.sum(torch.pow(q_targets - q_outputs, 2))



def get_parameters():
    input_size = 22528
    output_size = 9
    learning_rate = 3e-4 
    return input_size, output_size, learning_rate