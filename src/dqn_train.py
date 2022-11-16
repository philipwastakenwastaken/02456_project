import numpy as np
import torch
import wandb
import warnings
import time


from model import QNetwork, ReplayMemory
from eval import eval_model

warnings.filterwarnings("ignore")

def validate_model(env, dqnet, dev, use_wandb, episode_num=9):
    MODEL_VALIDATION_RATE = 100
    if (episode_num + 1) % MODEL_VALIDATION_RATE == 0:
        time_start = time.perf_counter()
        R_avg, frame_avg, count = validate(env, dqnet, dev)
        time_end = time.perf_counter()

        duration = time_end - time_start
        print(f'Avg. reward: {R_avg} Avg. frame count: {frame_avg} Avg. duration: {duration}')
        if use_wandb:
            wandb.log({'validate/avg_reward': R_avg,
                       'validate/avg_frame_count': frame_avg,
                       'validate/sims_per_val': count,
                       'validate/duration': duration}, 
                       commit=False)

def validate(env, dqnet, dev):
    TIME_CUTOFF = 20 # seconds
    
    R_avg = 0
    frame_avg = 0
    count = 0

    # Playing a set number of games can get expensive as the model gets better and games last longer.
    # Therefore, set a limit to how much time can be spent on simulating.
    start = time.perf_counter()
    while time.perf_counter() - start < TIME_CUTOFF:
        R, frame = eval_model(dqnet, env, dev)
        R_avg += R
        frame_avg += frame
        count += 1
    
    R_avg /= count
    frame_avg /= count

    return R_avg, frame_avg, count


def train_dq_model(dev, train_params, dqnet, target, model_path, use_wandb, checkpoint, env):
    # Train parameters
    num_episodes = train_params['num_episodes']
    episode_limit = train_params['episode_limit']
    gamma = train_params['gamma']  # discount rate
    val_freq = train_params['val_freq']  # validation frequency
    epsilon_start = train_params['epsilon_start']

    batch_size = train_params['batch_size']  # batch size
    tau = train_params['tau']  # target network update rate
    replay_memory_capacity = train_params['replay_memory_capacity']  # (?)
    prefill_memory = train_params['prefill_memory']  # (?)

    # Initialize policy- and target networks
    dqnet.to(torch.device(dev))
    target.to(torch.device(dev))
    target.load_state_dict(dqnet.state_dict())

    # Prefill replay memory with random actions
    replay_memory = ReplayMemory(replay_memory_capacity)
    if prefill_memory:
        print('< prefill replay memory >')
        s = env.reset()
        HEIGHT = s.shape[0]
        WIDTH = s.shape[1]
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
        frame_count = 0
        episode_start = 0
        validate_model(env, dqnet, dev, use_wandb)

        for i in range(episode_start, num_episodes):

            # initialize new episode
            s, ep_reward, ep_loss = env.reset(), 0, 0
            for j in range(episode_limit):

                # select action with epsilon-greedy strategy
                if np.random.rand() < epsilon:
                    a = env.action_space.sample()
                else:
                    with torch.no_grad():
                        a = dqnet(torch.from_numpy(s.reshape((1, 1, HEIGHT, WIDTH))).float().to(
                            torch.device(dev))).argmax().item()

                # perform action
                s1, r, d, _ = env.step(a)
                frame_count += 1

                # store experience in replay memory
                replay_memory.add(s, a, r, s1, d)

                # batch update
                if replay_memory.count() >= batch_size:

                    # sample batch from replay memory
                    batch = np.array(replay_memory.sample(batch_size))
                    ss, aa, rr, ss1, dd = batch[:, 0], batch[:,
                                                             1], batch[:, 2], batch[:, 3], batch[:, 4]
                    ss = np.stack(ss)
                    ss1 = np.stack(ss1)

                    # do forward pass of batch
                    dqnet.optimizer.zero_grad()
                    Q = dqnet(torch.from_numpy(ss.reshape(
                        (batch_size, 1, HEIGHT, WIDTH))).float().to(torch.device(dev)))

                    # use target network to compute target Q-values
                    with torch.no_grad():
                        Q1 = target(torch.from_numpy(ss1.reshape(
                            (batch_size, 1, HEIGHT, WIDTH))).float().to(torch.device(dev)))

                    # compute target for each sampled experience
                    q_targets = Q.clone()
                    for k in range(batch_size):
                        q_targets[k, aa[k]] = rr[k] + gamma * \
                            Q1[k].max().item() * (not dd[k])

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
                if d:
                    break

            # bookkeeping
            EPSILON_LOWER_LIMIT = 0.1
            epsilon *= num_episodes / (i / (num_episodes / 20) + num_episodes)  # decrease epsilon
            epsilon = max(epsilon, EPSILON_LOWER_LIMIT) # Set lower limit

            epsilons.append(epsilon)
            rewards.append(ep_reward)
            lengths.append(j+1)
            losses.append(ep_loss)
            mean_train_reward = np.mean(rewards[-val_freq:])
            if (i+1) % val_freq == 0:
                print('%5d mean training reward: %5.2f' %
                      (i+1, mean_train_reward))

            MODEL_SAVING_RATE = 10 # How often to save the model
            if (i + 1) % MODEL_SAVING_RATE == 0:
                dqnet.save(i, epsilon, model_path)
            
            validate_model(env, dqnet, dev, use_wandb, i)

            # This is pretty ugly... but making a fully fledged logger is pretty time consuming
            if use_wandb:
                wandb.log({'mean_train_reward': mean_train_reward,
                           'frame_count': frame_count,
                           'epsilon': epsilon})

        print('done')

        # Save network weights
        print(model_path)
        dqnet.save(i, epsilon, model_path)
        print('Saved model')

        return rewards, lengths, losses, epsilons

    except KeyboardInterrupt:
        print('interrupt')
