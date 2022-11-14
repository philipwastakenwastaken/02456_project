import gym
from skimage.transform import resize
import numpy as np
import os
import constant
from abc import ABC, abstractmethod

class BaseWrapper(gym.wrapper, ABC):

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    def binarize(self, s):
        s[s>constant.BINARY_THRESHOLD] = 1
        s[s<=constant.BINARY_THRESHOLD] = 0
        return s

class CropWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        s = s[6:170,5:-5]
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[6:170,5:-5]
        return s


class StretchWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        s = resize(s, constant.RESIZE_SHAPE, anti_aliasing=False)
        s[s>0.3]=1
        s[s<=0.3]=0
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = resize(s, constant.RESIZE_SHAPE, anti_aliasing=False)
        s[s>0.3]=1
        s[s<=0.3]=0
        return s


class ResizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(5)

    def step(self, action):
        s, r, done, info = self.env.step(action)
        s = s[6:170,5:-5]
        s = resize(s, constant.RESIZE_SHAPE, anti_aliasing=False)
        s[s>0.3]=1
        s[s<=0.3]=0
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[6:170,5:-5]
        s = resize(s, constant.RESIZE_SHAPE, anti_aliasing=False)
        s[s>0.3]=1
        s[s<=0.3]=0
        return s
