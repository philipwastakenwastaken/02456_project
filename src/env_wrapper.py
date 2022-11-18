import gym
from skimage.transform import resize
import numpy as np
import os
from skimage import io
from matplotlib import pyplot as plt

ACTION_SPACE_SIZE = 5
RESIZE_DIM = (72, 72)


class NopWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        s[s < 76] = 0
        s[s >= 76] = 255
        if info['lives'] == 2:
            done = True
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        s[s < 76] = 0
        s[s >= 76] = 255
        for i in range(65):
            self.env.step(0)
        return s


class CropWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[6:170, 5:-5]
        s[s < 76] = 0
        s[s >= 76] = 255
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[6:170, 5:-5]
        s[s < 76] = 0
        s[s >= 76] = 255
        return s


class StretchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = resize(s, RESIZE_DIM, anti_aliasing=False)
        s[s > 0.135] = 1
        s[s <= 0.135] = 0
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = resize(s, RESIZE_DIM, anti_aliasing=False)
        s[s > 0.135] = 1
        s[s <= 0.135] = 0
        return s


class ResizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[6:170, 5:-5]
        s = resize(s, RESIZE_DIM, anti_aliasing=False)
        s[s > 0.135] = 1
        s[s <= 0.135] = 0
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[6:170, 5:-5]
        s = resize(s, RESIZE_DIM, anti_aliasing=False)
        s[s > 0.135] = 1
        s[s <= 0.135] = 0
        return s

class ResizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = s[6:170, 5:-5]
        s = resize(s, RESIZE_DIM, anti_aliasing=False)
        s[s > 0.135] = 1
        s[s <= 0.135] = 0
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[6:170, 5:-5]
        s = resize(s, RESIZE_DIM, anti_aliasing=False)
        s[s > 0.135] = 1
        s[s <= 0.135] = 0
        return s

ACTION_SPACE_SIZE = 5
BINARY_THRESHOLD = 0.135
CROP_RANGE = 84
RESIZE_DIM_NEW = (110, CROP_RANGE)

class DMWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

    def binarize(self, s):
        s[s > BINARY_THRESHOLD] = 1
        s[s <= BINARY_THRESHOLD] = 0
        return s


    def step(self, action):
        s, r, done, info = self.env.step(action)
        if info['lives'] == 2:
            done = True
        s = resize(s, RESIZE_DIM_NEW, anti_aliasing=False)
        s = s[3:CROP_RANGE+3, :]

        self.binarize(s)

        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        print(s.shape)
        for _ in range(65):
            self.env.step(0)
        s = resize(s, RESIZE_DIM_NEW, anti_aliasing=False)
        s = s[3:CROP_RANGE+3, :]
        self.binarize(s)
        return s