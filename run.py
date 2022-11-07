import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from skimage import io
from model import QNetwork, get_parameters


<<<<<<< HEAD
# env = gym.make('ALE/Asterix-v5',full_action_space=False, obs_type='grayscale',render_mode='human')
env = gym.make('ALE/Asterix-v5',full_action_space=False, obs_type='grayscale')
=======
env = gym.make('ALE/MsPacman-v5',full_action_space=False, obs_type='grayscale',render_mode='human')
>>>>>>> 1b950d5ef6d8c4813022d3458b1c290de1f7ee36

n_inputs, n_outputs, learning_rate = get_parameters() 
qnet = QNetwork(n_inputs, n_outputs, learning_rate)
qnet.load_state_dict(torch.load('qnet.pt'))

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

rewardList = []

for i in range(200):
    s = env.reset()

    R = 0
    for j in range(20000):
        a = qnet(torch.from_numpy(s.reshape((1, 1, 210, 160))).float()).argmax().item()
        s, r, done, _ = env.step(a)

        R += r

        if done:
            print("Died at frame:", j)
            break

    rewardList.append(R)

    print("Total reward in round {} is {}.".format(i, R))


plt.plot(rewardList)
plt.xlabel('Round')
plt.ylabel('mean training reward')
plt.show()

