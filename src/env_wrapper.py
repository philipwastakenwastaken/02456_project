import gym
from skimage.transform import resize
import numpy as np
import os


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
        s = resize(s, (72,72), anti_aliasing=False)
        s[s>0.3]=1
        s[s<=0.3]=0
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = resize(s, (72,72), anti_aliasing=False)
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
        s = resize(s, (72,72), anti_aliasing=False)
        s[s>0.3]=1
        s[s<=0.3]=0
        return s, r, done, info

    def reset(self):
        s = self.env.reset()
        for i in range(65):
            self.env.step(0)
        s = s[6:170,5:-5]
        s = resize(s, (72,72), anti_aliasing=False)
        s[s>0.3]=1
        s[s<=0.3]=0
        return s 
