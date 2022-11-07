import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
from skimage import io
from env_factory import make_env
import warnings
warnings.filterwarnings("ignore")

def eval_model(dev, eval_params, dqnet, env_params):
    env = make_env(env_params, render_mode='human')
    s = env.reset()

    R = 0
    for i in range(2000):
        a = dqnet(torch.from_numpy(s.reshape((1,1,210,160))).float()).argmax().item()
        s, r, done, _ = env.step(a)

        R += r

        if done:
            print("Died at frame:",i)
            break

    print("Total reward", R)
    return R
