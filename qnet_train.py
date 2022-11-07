import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from skimage import io
from model import QNetwork, get_parameters

env = gym.make('ALE/Asterix-v5',full_action_space=False, obs_type='grayscale')

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"



# train Q-network
num_episodes = 100
episode_limit = 300
gamma = 0.9 # discount rate
val_freq = 1 # validation frequency
epsilon_start = 1.0

n_inputs, n_outputs, learning_rate = get_parameters() 
qnet = QNetwork(n_inputs, n_outputs, learning_rate)
qnet.to(torch.device(dev))

rewardsList = []

try:
    epsilon = epsilon_start
    rewards, lengths, losses, epsilons = [], [], [], []
    print('start training')
    for i in range(num_episodes):
        # init new episode
        s, ep_reward, ep_loss = env.reset(), 0, 0
        for j in range(episode_limit):
            
            # 1. do foward pass of current state to compute Q-values for all actions
            qnet.optimizer.zero_grad()
            Q = qnet(torch.from_numpy(s.reshape((1,1,210,160))).float().to(torch.device(dev)))
            
            # 2. select action with epsilon-greedy strategy
            a = Q.argmax().item() if np.random.rand() > epsilon else env.action_space.sample()
            s1, r, done, _ = env.step(a)
            
            # 3. do forward pass for the next state
            with torch.no_grad():
                Q1 = qnet(torch.from_numpy(s.reshape((1,1,210,160))).float().to(torch.device(dev)))
            
            # 4. set Q-target
            q_target = Q.clone()
            q_target[a] = r + gamma * Q1.max().item() * (not done)
            
            # 5. update network weights
            loss = qnet.loss(Q, q_target)
            loss.backward()
            qnet.optimizer.step()
            
            # 6. bookkeeping
            s = s1
            ep_reward += r
            ep_loss += loss.item()
            if done: break
        
        # bookkeeping
<<<<<<< HEAD
        epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
        epsilon -= 0.0005
=======
        #epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
        #epsilon -= 0.0001
>>>>>>> 1b950d5ef6d8c4813022d3458b1c290de1f7ee36
        epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(j+1); losses.append(ep_loss)
        rewardsList.append(np.mean(rewards[-val_freq:]))
        if (i+1) % val_freq == 0: print('{:5d} mean training reward: {:5.2f}'.format(i+1, np.mean(rewards[-val_freq:])))
    print("Mean reward is {}".format(np.mean(rewardsList)))
    plt.plot(rewardsList)
    plt.xlabel('Episode')
    plt.ylabel('mean training reward')
    plt.show()
    print('done')
except KeyboardInterrupt:
    print('interrupt')


torch.save(qnet.state_dict(), 'qnet.pt',_use_new_zipfile_serialization=False)