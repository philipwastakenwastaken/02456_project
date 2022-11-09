import gym
from skimage.transform import resize
import numpy as np

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

class GridifyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = gym.spaces.Discrete(5)
        
        self.track = np.loadtxt('map_track.txt', dtype='i', delimiter=',')
        self.pacman_col = 167
        self.g1_col = 110
        self.g2_col = 131
        self.g3_col = 132
        self.g4_col = 151


    def where(self, s, r):
        w = 18/s.shape[1] # 105
        h = 14/s.shape[0] # 80
        

        entities = [None,None,None,None,None]
        pellet = None
        for x in range(s.shape[1]):
            for y in range(s.shape[0]):
                    
                if not entities[0] and s[y,x]==self.pacman_col:
                    entities[0]=(int(y*h),int(x*w))
                    if r==10:
                        pellet=(int(y*h),int(x*w))
                        
                elif not entities[1] and s[y,x]==self.g1_col:
                         entities[1]=(int(y*h),int(x*w))
                        
                elif not entities[2] and s[y,x]==self.g2_col:
                         entities[2]=(int(y*h),int(x*w))
                        
                elif not entities[3] and s[y,x]==self.g3_col:
                         entities[3]=(int(y*h),int(x*w))
                        
                elif not entities[4] and s[y,x]==self.g4_col:
                         entities[4]=(int(y*h),int(x*w))
        return entities, pellet

    def step(self, action):
        s, r, done, info = self.env.step(4)
        s = s[4:166,5:-5]
        s = np.delete(s, list(range(0, s.shape[0], 2)), axis=0)
        s = np.delete(s, list(range(0, s.shape[1], 2)), axis=1)
        s = np.delete(s, list(range(0, s.shape[0], 2)), axis=0)
        s = np.delete(s, list(range(0, s.shape[1], 2)), axis=1)

        position = [None,None,None,None,None]
        pmap = self.track.copy()

        for i in range(200):      
            smap = self.track.copy()
            entities, pellet = self.where(s,r)
            if pellet != None:
                pmap[pellet]=3
            
            # Don't update position if entities are invisible
            for i, ent in enumerate(entities):
                if ent != None:
                    position[i]=ent
                    
            # Update entities if visible
            for ent in position[0:1]:
                if ent != None:
                    here = smap[ent[0],ent[1]]
                    smap[ent[0],ent[1]]=3
                        
            for ent in position[1:]:
                if ent != None:
                    here = smap[ent[0],ent[1]]
                    smap[ent[0],ent[1]]=2
            
        return smap, r, done, info

    def reset(self):
        s = self.env.reset()[4:166,5:-5]
        for i in range(65):
            self.env.step(0)
        s = np.delete(s, list(range(0, s.shape[0], 2)), axis=0)
        s = np.delete(s, list(range(0, s.shape[1], 2)), axis=1)
        s = np.delete(s, list(range(0, s.shape[0], 2)), axis=0)
        s = np.delete(s, list(range(0, s.shape[1], 2)), axis=1)
        return self.track