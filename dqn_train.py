import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from skimage import io
from model import QNetwork, ReplayMemory,get_parameters


# Initialize environment
env = gym.make('ALE/Asterix-v5',full_action_space=False, obs_type='grayscale')

# Use GPU if possible
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

# Train parameters 
num_episodes = 20 # number of games
episode_limit = 300 # game length (in frames)
gamma = 0.9 # discount rate
val_freq = 1 # validation frequency
epsilon_start = 1.0 # amount of randomness

batch_size = 8 # batch size
tau = 0.01 # target network update rate
replay_memory_capacity = 100 # (?)
prefill_memory = True #(?)
val_freq = 1 # after how many training episodes should we show the result

# Initialize policy- and target networks
n_inputs, n_outputs, learning_rate = get_parameters() 
dqnet = QNetwork(n_inputs, n_outputs, learning_rate)
dqnet.to(torch.device(dev))
target = QNetwork(n_inputs, n_outputs, learning_rate)
target.to(torch.device(dev))
target.load_state_dict(dqnet.state_dict())

# Prefill replay memory with random actions
replay_memory = ReplayMemory(replay_memory_capacity)
if prefill_memory:
    print('< prefill replay memory >')
    s = env.reset()
    while replay_memory.count() < replay_memory_capacity:
        a = env.action_space.sample()
        s1, r, d, _ = env.step(a)
        replay_memory.add(s, a, r, s1, d)
        s = s1 if not d else env.reset()

# Training loop
try:
    print('< start training >')
    epsilon = 1.0
    rewards, lengths, losses, epsilons = [], [], [], []
    for i in range(num_episodes):
        
        # initialize new episode
        s, ep_reward, ep_loss = env.reset(), 0, 0
        for j in range(episode_limit):
            
            # select action with epsilon-greedy strategy
            if np.random.rand() < epsilon:
                a = env.action_space.sample()
            else:
                with torch.no_grad():
                    a = dqnet(torch.from_numpy(s.reshape((1,1,210,160))).float().to(torch.device(dev))).argmax().item()
            
            # perform action
            s1, r, d, _ = env.step(a)
            
            # store experience in replay memory
            replay_memory.add(s, a, r, s1, d)
            
            # batch update
            if replay_memory.count() >= batch_size:
                
                # sample batch from replay memory
                batch = np.array(replay_memory.sample(batch_size))
                ss, aa, rr, ss1, dd = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
                ss = np.stack(ss)
                ss1 = np.stack(ss1)
                
                # do forward pass of batch
                dqnet.optimizer.zero_grad()
                Q = dqnet(torch.from_numpy(ss.reshape((batch_size,1,210,160))).float().to(torch.device(dev)))

                # use target network to compute target Q-values
                with torch.no_grad():
                    Q1 = target(torch.from_numpy(ss1.reshape((batch_size,1,210,160))).float().to(torch.device(dev)))
                
                # compute target for each sampled experience
                q_targets = Q.clone()
                for k in range(batch_size):
                    q_targets[k, aa[k]] = rr[k] + gamma * Q1[k].max().item() * (not dd[k])
                
                # update network weights
                loss = dqnet.loss(Q, q_targets)
                loss.backward()
                dqnet.optimizer.step()
                
                # update target network parameters from policy network parameters
                target.update_params(dqnet.state_dict(), tau)
            
            else:
                loss = 0
            
            # bookkeeping
            s = s1
            ep_reward += r
            ep_loss += loss.item()
            if d: break

        # bookkeeping
        epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
        epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(j+1); losses.append(ep_loss)
        if (i+1) % val_freq == 0: print('%5d mean training reward: %5.2f' % (i+1, np.mean(rewards[-val_freq:])))
    
    print('done')

except KeyboardInterrupt:
    print('interrupt')

# Save network weights
torch.save(dqnet.state_dict(), 'dqnet.pt',_use_new_zipfile_serialization=False)