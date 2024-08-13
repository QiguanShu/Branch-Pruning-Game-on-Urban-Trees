#!/usr/bin/env python
# coding: utf-8

# # 1. Import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os

import gym
from gym import spaces

from logger import logger

# # 2. Reset
# ## 2.1 define basic functions
def crownExpansion(voxData):
    addVox = np.array([[[False for _ in range(voxData.shape[2])] for _ in range(voxData.shape[1])] for _ in range(voxData.shape[0])], dtype=bool)
    for x in range(1,voxData.shape[0]-1):
        for y in range(1,voxData.shape[1]-1):
            for z in range(0,voxData.shape[2]-1):
                if voxData[x,y,z] == True:
                    #addVox[x-1,y-1,z] = True
                    addVox[x-1,y,z] = True
                    #addVox[x-1,y+1,z] = True
                    #addVox[x-1,y-1,z+1] = True
                    #addVox[x-1,y,z+1] = True
                    #addVox[x-1,y+1,z+1] = True
                    addVox[x,y-1,z] = True
                    #addVox[x,y,z] = True
                    addVox[x,y+1,z] = True
                    #addVox[x,y-1,z+1] = True
                    addVox[x,y,z+1] = True
                    #addVox[x,y+1,z+1] = True
                    #addVox[x+1,y-1,z] = True
                    addVox[x+1,y,z] = True
                    #addVox[x+1,y+1,z] = True
                    #addVox[x+1,y-1,z+1] = True
                    #addVox[x+1,y,z+1] = True
                    #addVox[x+1,y+1,z+1] = True
    return np.logical_or(voxData, addVox)


def voxThinning (voxData, rate):
    for x in range(0,voxData.shape[0]):
        for y in range(0,voxData.shape[1]):
            for z in range(0,voxData.shape[2]):
                if voxData[x,y,z] == True:
                    if np.random.rand() < rate:
                        voxData[x,y,z] = False  
    return voxData


def voxRaise (voxData, height):
    # Get the z-coordinates of the voxel values
    z_coords = np.where(voxData)[2]
    # Find the minimum z-coordinate
    min_z = np.min(z_coords)
    # Set the corresponding values to zero
    voxData[:,:,:(min_z+height)] = 0
    return voxData


def voxReductionTop (voxData, dist):
    # Get the z-coordinates of the voxel values
    z_coords = np.where(voxData)[2]
    # Find the maximum z-coordinate
    max_z = np.max(z_coords)
    # Set the corresponding values to zero
    voxData[:,:,(max_z-dist+1):] = 0
    return voxData


def voxReductionWest (voxData, dist):
    # Get the x-coordinates of the voxel values
    x_coords = np.where(voxData)[0]
    # Find the minimum x-coordinate
    min_x = np.min(x_coords)
    # Set the corresponding values to zero
    voxData[:(min_x+dist),:,:] = 0
    return voxData


def voxReductionEast (voxData, dist):
    # Get the x-coordinates of the voxel values
    x_coords = np.where(voxData)[0]
    # Find the maximum x-coordinate
    max_x = np.max(x_coords)
    # Set the corresponding values to zero
    voxData[(max_x-dist+1):,:,:] = 0
    return voxData


def voxReductionSouth (voxData, dist):
    
    # Get the y-coordinates of the voxel values
    y_coords = np.where(voxData)[1]
    # Find the minimum y-coordinate
    min_y = np.min(y_coords)
    # Set the corresponding values to zero
    voxData[:,:(min_y+dist),:] = 0
    return voxData


def voxReductionNorth (voxData, dist):
    # Get the y-coordinates of the voxel values
    y_coords = np.where(voxData)[1]
    # Find the maximum y-coordinate
    max_y = np.max(y_coords)
    # Set the corresponding values to zero
    voxData[:,(max_y-dist+1):,:] = 0
    return voxData


def distToCenter (voxData):
    # Collect voxel centers with values greater than 0.05
    voxelCenters = np.argwhere(voxData == True)
    meanCenter = np.mean(voxelCenters,axis=0)
    center = np.array([meanCenter[0], meanCenter[1], meanCenter[2]])
    
    dists = np.zeros_like(voxData, dtype=np.float32)
    for i in range(voxData.shape[0]):
        for j in range(voxData.shape[1]):
            for k in range(voxData.shape[2]):
                if voxData[i,j,k] == True:
                    voxel_center = np.array([i, j, k])
                    dists[i, j, k] = np.linalg.norm(voxel_center - center)                
    return dists, np.max(dists)


def voxShrink (voxData, dist):
    distMatrix, maxDist = distToCenter (voxData)
    for i in range(voxData.shape[0]):
        for j in range(voxData.shape[1]):
            for k in range(voxData.shape[2]):
                if (voxData[i,j,k] == True) and (distMatrix[i,j,k] > (maxDist-dist)):
                    voxData[i,j,k] = False    
    return voxData


def plot_LAD (preLAD, fileName: str):
    # plot the LAD in voxels
    # Create a 3D figure
    fig = plt.figure(figsize=(12, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Create a mask based on the solid voxels
    mask = preLAD
    
    # Create a color map, with alpha (transparency) based on normalized values
    colors = np.zeros((*preLAD.shape, 4), dtype=object)
    colors[..., :3] = [0, 1, 0]  # Set the RGB channels
    colors[..., 3] = 0.75  # Set the alpha channel

    # Voxel plot
    ax.set_aspect('equal')  # Aspect ratio is 1:1:1 in data space
    ax.voxels(mask, facecolors=colors, edgecolors="black", linewidth=0.2)

    #plt.show()
    plt.savefig(f'savefig_bin/{fileName}.png')


# ## 2.2 main function
def setInitialLAD (voxDim):
    voxData = np.array([[[False for _ in range(voxDim[2])] for _ in range(voxDim[1])] for _ in range(voxDim[0])], dtype=bool)

    # create one seed point to grow the target crown
    seeds = [[voxDim[0]/2, voxDim[1]/2, 3]]
    
    # initiate seeds in the voxData
    for start in seeds:
        voxData[int(start[0]),int(start[1]),int(start[2])] = True
    
    # expansion the crown from the starts for N rounds
    n = 3
    for _ in range(n):
        voxData = crownExpansion(voxData)
    
    return voxData


def setTargetLAD (voxDim):
    voxData = np.array([[[False for _ in range(voxDim[2])] for _ in range(voxDim[1])] for _ in range(voxDim[0])], dtype=bool)

    # create several seed points to grow the target crown
    seeds = [[6,9,13],[8,9,19],[10,7,15]]
    
    # initiate seeds in the voxData
    for start in seeds:
        voxData[start[0],start[1],start[2]] = True
        #print(f'seed position {start} is set with a targetLAD {startLAD}')
    
    n = 8
    for _ in range(n):
        voxData = crownExpansion(voxData)
    
    voxData = voxShrink(voxData, 3.5)
    voxData = crownExpansion(voxData)
    
    return voxData


def calculateScore_IoU (targetLAD, predictLAD):
    if targetLAD.shape != predictLAD.shape:
        raise ValueError("Both arrays must have the same shape.")

    # Calculate the IoU score
    score = np.logical_and(targetLAD, predictLAD).sum() / np.logical_or(targetLAD, predictLAD).sum() * 100
    
    return score


def initiatePrunGame(Ep,Rd):
    global projektNo
    projektNo = 0
    projektNo += 1
    
    global voxSize
    voxSize = 0.8
    global voxDim
    voxDim = [20, 20, 26] # total voxel numbers along x, y and z axis
    
    # set a initial LAD values for each voxel
    global predictLAD
    predictLAD = setInitialLAD(voxDim)
    
    # set a target LAD values for each voxel
    global targetLAD
    targetLAD = setTargetLAD(voxDim)

    global score
    score = calculateScore_IoU(targetLAD, predictLAD)
    
    # Check if the folder already exists; if not, create it
    if not os.path.exists("savefig_bin"):
        os.makedirs("savefig_bin")
    #else:
        #print(f"Folder 'savefig_simp' already exists.")

    # plot current QSM model and voxel Model
    if Ep % 200 ==0 or Ep==999:
        fileNameLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd{Rd:02d}_Sc{score:.2f}_LAD"
        plot_LAD(predictLAD, fileNameLAD)
        fileNameTarLAD = f"targetLAD"
        plot_LAD(targetLAD, fileNameTarLAD)
    
    state_initiate = predictLAD # this is the first state of the episode
    state_initiate.astype(bool)
    
    return state_initiate, score


# # 3. Step
def runPrunSimulation (action, Ep, Rd, randTreeNo, last_reward):
    #try:
        done = False

        # updata voxel based on action input
        global predictLAD
            
        logger.info(f"++ Chosen action : [{action[0]},{action[1]}]")
        if action[0]==0: # THINNING
            LAD_updated = voxThinning(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        if action[0]==0: # RAISING
            LAD_updated = voxRaise(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        elif action[0]==1: # REDUCTION_EAST
            LAD_updated = voxReductionEast(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        elif action[0]==2: # REDUCTION_SOUTH
            LAD_updated = voxReductionSouth(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        elif action[0]==3: # REDUCTION_WEST
            LAD_updated = voxReductionWest(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        elif action[0]==4: # REDUCTION_NORTH
            LAD_updated = voxReductionNorth(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        elif action[0]==5: # REDUCTION_TOP
            LAD_updated = voxReductionTop(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        elif action[0]==6: # Topping
            LAD_updated = voxShrink(predictLAD, action[1])
            LAD_updated = crownExpansion(LAD_updated)
        elif action[0]==7: # NOACTION
            LAD_updated = crownExpansion(predictLAD)
        else:               # ENDING
            LAD_updated = predictLAD        
            done = True
            
        logger.info(f"++ predictLAD size after cutting: {LAD_updated.sum()}")
            
        if Rd == 30:
            done = True
        
        reward_IoU_cur = calculateScore_IoU(targetLAD, LAD_updated)
        reward_IoU = reward_IoU_cur - last_reward
        last_reward = reward_IoU_cur
        
        if Rd > 10:
            reward_IoU = reward_IoU - 0.2*(Rd-10) # if a high score can not be reached by 10 years, each extra year will cost 0.2 point reduction
        reward = reward_IoU
        
        if LAD_updated.sum() == 0:
            reward = -100
            done = True
        else:
            reward += 0.1

        # plot current QSM model and voxel Model
        # the image name should follow "TreeXX_EpXX_Rd00_ScXXX_QSM/LAD"
        if Ep % 200 ==0 or Ep==999:
            fileNameLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd{Rd:02d}_Sc{reward:.2f}_LAD"
            plot_LAD(LAD_updated, fileNameLAD)
        
        predictLAD = LAD_updated
        next_state = LAD_updated
        next_state.astype(bool)

        return next_state, reward, done, last_reward
    
class PrunEnvWrapper(gym.Env):
    def __init__(self,):
        # Tree Number 
        self.randTreeNo = ""
        
        # State size, 1D vector flatten from 3D voxel
        self.dimension = 20*20*26
        #self.observation_space = spaces.Box(0, 1, shape=(self.dimension,), dtype=np.float32)
        self.observation_space = spaces.Box(0, 1, shape=((20,20,26)), dtype=np.float32)
        
        # Action space
        self.num_actions = 10
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_actions), 
            spaces.Box(0, 10, shape=(1,), dtype=np.float32)
        ))
        
        self.episodes = 0
        self.episode_steps = 0 # Rd
        self.state = None
        self.last_reward = 0.0
        
    def step(self, action):
        next_state, reward, done, last_reward = runPrunSimulation(action, self.episodes, self.episode_steps, self.randTreeNo, self.last_reward)
        self.last_reward = last_reward
        #next_state, reward, done = runPrunSimulation(action)
        if next_state is not None:
            # self.state = next_state
            next_state = next_state.astype(float) #next_state.reshape(-1)
        else:
            next_state = self.observation_space.sample() #.reshape(-1)
            
        return next_state, reward, done, {}
    
    # current state
    def reset(self):
        #state, randTreeNo = initiatePrunGame(self.episodes, self.episode_steps)
        state, score = initiatePrunGame(self.episodes, self.episode_steps)
        self.last_reward = score
        #self.randTreeNo = randTreeNo
        if state is not None:
            # self.state = state
            state = state.astype(float) #.reshape(-1)
        return state