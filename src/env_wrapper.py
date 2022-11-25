import gym
from skimage.transform import resize
import numpy as np
import os
from skimage import io
from matplotlib import pyplot as plt

ACTION_SPACE_SIZE = 5

class NopWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        return s/255, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        return s/255


class CropWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[3:170, 5:-5]
        return s/255, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[3:170, 5:-5]
        return s/255


class Scale120Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.resize_dim = (120,120)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s

class Scale84Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.resize_dim = (84,84)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s

class Scale72Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.resize_dim = (72,72)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s
    
class Scale60Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.resize_dim = (60,60)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s

class Scale48Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.resize_dim = (48,48)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[3:170, 5:-5]
        s = resize(s, self.resize_dim, anti_aliasing=False)
        return s