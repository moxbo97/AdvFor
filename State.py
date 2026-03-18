import numpy as np
import sys
import cv2
action_map = {0:[0,0,0],1:[0,0,1],2:[0,0,-1],
                3:[0,1,0],4:[0,1,1],5:[0,1,-1],
                6:[0,-1,0],7:[0,-1,1],8:[0,-1,-1],
                9:[1,0,0],10:[1,0,1],11:[1,0,-1],
                12:[1,1,0],13:[1,1,1],14:[1,1,-1],
                15:[1,-1,0],16:[1,-1,1],17:[1,-1,-1],
                18:[-1,0,0],19:[-1,0,1],20:[-1,0,-1],
                21:[-1,1,0],22:[-1,1,1],23:[-1,1,-1],
                24:[-1,-1,0],25:[-1,-1,1],26:[-1,-1,-1]}
gray_action_map = {0:[0,0,0],1:[1,1,1],2:[-1,-1,-1]}
# scale_action_map = {0:[1,1,1],1:[1.05,1.05,1.05],2:[0.95,0.95,0.95]}
# scale_action_map = {0:[1,1,1],1:[1,1,1.03],2:[1,1,0.97],
                # 3:[1,1.03,1],4:[1,1.03,1.03],5:[1,1.03,0.97],
                # 6:[1,0.97,1],7:[1,0.97,1.03],8:[1,0.97,0.97],
                # 9:[1.03,1,1],10:[1.03,1,1.03],11:[1.03,1,0.97],
                # 12:[1.03,1.03,1],13:[1.03,1.03,1.03],14:[1.03,1.03,0.97],
                # 15:[1.03,0.97,1],16:[1.03,0.97,1.03],17:[1.03,0.97,0.97],
                # 18:[0.97,1,1],19:[0.97,1,1.03],20:[0.97,1,0.97],
                # 21:[0.97,1.03,1],22:[0.97,1.03,1.03],23:[0.97,1.03,0.97],
                # 24:[0.97,0.97,1],25:[0.97,0.97,1.03],26:[0.97,0.97,0.97]}
class State():
    def __init__(self, size, move_range):
        self.state = np.zeros(size, dtype=np.float32)
        self.move_range = np.float32(move_range)
        self.image = np.zeros(size, dtype=np.float32)
        
    def reset(self, x):
        self.image = x.copy()
        self.state = np.clip(x/255., a_min=0., a_max=1.)
        #self.state[:,3,:,:] = action
        
    def step(self, act, sigma):
        move_map = np.ones(self.image.shape, dtype = np.float32)
        act = act[:,np.newaxis,:,:]
        tmp = np.ones(self.image.shape, dtype = np.float32)
        tmp[:,0,:,:] = act[:,0,:,:]
        tmp[:,1,:,:] = act[:,0,:,:]
        tmp[:,2,:,:] = act[:,0,:,:]
        move_map[tmp==0] -=sigma 
        move_map[tmp==2] +=sigma
        self.image *= move_map
        self.image = np.clip(self.image, a_min=0., a_max=255)
        self.state = self.image/255.
        return move_map
