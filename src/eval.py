import torch
import time
import warnings
import numpy as np
import scipy.stats as st

warnings.filterwarnings("ignore")

    


def eval_model(dqnet, env, dev):
    s = env.reset()
    HEIGHT = s.shape[0]
    WIDTH = s.shape[1]

    done = False
    frame = 1
    R = 0
    while not done:
        a = dqnet(torch.from_numpy(s.reshape((1, 1, HEIGHT, WIDTH))).float().to(
            torch.device(dev))).argmax().item()
        s, r, done, _ = env.step(a)

        R += r
        frame += 1

    return R, frame



class ValidationRunInfo:
    def __init__(self):
        self.frame_counts = []
        self.rewards = []
        self.terminated = False
    
    def add_run(self, reward, frame_count):
        if self.count() == 0:
            self.start_time = time.perf_counter()

        self.rewards.append(reward)
        self.frame_counts.append(frame_count)
    
    def reward_error(self):
        return st.t.interval(0.95, 
                             self.count() - 1, 
                             loc=np.mean(self.rewards), 
                             scale=st.sem(self.rewards))

    
    def done(self, total_time):
        self.frame_counts = np.array(self.frame_counts)
        self.rewards = np.array(self.rewards)

        self.reward_mean = np.mean(self.rewards)
        self.frame_count_mean = np.mean(self.frame_counts)

        # 95% conf. interval
        self.reward_error = self.reward_error()
        
        self.actual_duration = total_time
        self.terminated = True
    
    def count(self):
        return len(self.rewards)

class ModelValidator:
    def __init__(self, model, env, dev, time_cutoff=20):
        self.model = model
        self.env = env
        self.dev = dev
        self.time_cutoff = time_cutoff
        self.run_info = ValidationRunInfo()
    
    # Simulate games until time cutoff has been reached
    def run(self):
        start = time.perf_counter()
        while time.perf_counter() - start < self.time_cutoff:
            R, frame = eval_model(self.model, self.env, self.dev)
            self.run_info.add_run(R, frame)
        total_time = time.perf_counter() - start
        
        self.run_info.done(total_time)
        
        return self.run_info
