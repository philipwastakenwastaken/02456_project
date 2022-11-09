import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from skimage import io
from model import QNetwork, ReplayMemory
from env_factory import make_env
import warnings
warnings.filterwarnings("ignore")

def train_dq_model(dev, train_params, dqnet, target, model_path, env_params):
    # Initialize environment
    env = make_env(env_params)

    # Train parameters
    num_episodes = train_params['num_episodes']
    episode_limit = train_params['episode_limit']
    gamma = train_params['gamma'] # discount rate
    val_freq = train_params['val_freq'] # validation frequency
    epsilon_start = train_params['epsilon_start']

    batch_size = train_params['batch_size'] # batch size
    tau = train_params['tau'] # target network update rate
    replay_memory_capacity = train_params['replay_memory_capacity'] # (?)
    prefill_memory = train_params['prefill_memory'] #(?)

    # Initialize policy- and target networks
    dqnet.to(torch.device(dev))
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
        epsilon = epsilon_start
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
            #epsilon *= num_episodes/(i/(num_episodes/20)+num_episodes) # decrease epsilon
            epsilon -= 0.001
            epsilons.append(epsilon); rewards.append(ep_reward); lengths.append(j+1); losses.append(ep_loss)
            if (i+1) % val_freq == 0: print('%5d mean training reward: %5.2f' % (i+1, np.mean(rewards[-val_freq:])))

        print('done')
        return rewards, lengths, losses, epsilons

    except KeyboardInterrupt:
        print('interrupt')

    # Save network weights
    torch.save(dqnet.state_dict(), model_path, _use_new_zipfile_serialization=False)