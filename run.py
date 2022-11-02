import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from skimage import io
from model import QNetwork, get_parameters


env = gym.make('ALE/MsPacman-v5',full_action_space=False, obs_type='grayscale',render_mode='human')

n_inputs, n_outputs, learning_rate = get_parameters() 
qnet = QNetwork(n_inputs, n_outputs, learning_rate)
qnet.load_state_dict(torch.load('qnet.pt'))

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  


s = env.reset()

R = 0
for i in range(2000):
    a = qnet(torch.from_numpy(s.reshape((1,1,210,160))).float()).argmax().item()
    s, r, done, _ = env.step(a)

    R += r

    if done: 
        print("Died at frame:",i)
        break

print("Total reward", R)
