#!/usr/bin/env python
# coding: utf-8

# # 1. Import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
pd.options.mode.chained_assignment = None 
import gym
from gym import spaces

import os
from logger import logger


# # 2. Reset
# ## 2.1 define basic functions

def crownExpansion(voxData, seed, force):
    x, y, z = seed
    adjacent_voxels = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                nx, ny, nz = x + i, y + j, z + k
                if (0 <= nx < len(voxData) and
                    0 <= ny < len(voxData[0]) and
                    0 <= nz < len(voxData[0][0]) and
                    (i, j, k) != (0, 0, 0)):
                    if(voxData[nx][ny][nz] == 0):
                        addLAD = np.float32(np.random.uniform(0, force))
                        voxData[nx][ny][nz] += addLAD
                        #print(f'seed position {[nx, ny, nz]} is add {addLAD} to its targetLAD')
    return voxData


def generateSeedsTar(voxDim, seedNum, z_min):
    # changes start
    """ old version without z_max limit
    all_locations = np.array([(x, y, z) for x in range(voxDim[0]) for y in range(voxDim[1]) for z in range(z_min, voxDim[2])])
    """
    # new version add a margin restriction for the seed positions
    margin = 5
    all_locations = np.array([(x, y, z) for x in range(margin, voxDim[0]-margin) for y in range(margin, voxDim[1]-margin) for z in range(z_min, voxDim[2]-margin)])
    # changes end
    indices = np.random.choice(all_locations.shape[0], seedNum, replace=False)
    return all_locations[indices]


def generateSeedsInit(voxDim, z_max):
    # changes start
    """ old version without z_max limit
    all_locations = np.array([(x, y, z) for x in range(voxDim[0]) for y in range(voxDim[1]) for z in range(z_min, voxDim[2])])
    """
    # new version add a margin restriction for the seed positions
    margin = 5
    all_locations = np.array([(x, y, z) for x in range(margin, voxDim[0]-margin) for y in range(margin, voxDim[1]-margin) for z in range(margin, z_max)])
    # changes end
    indices = np.random.choice(all_locations.shape[0], 1, replace=False)
    return all_locations[indices]


def rotation_matrix(A, angle):
    """
    Returns the rotation matrix for the given axis A and angle (in radians).

    Parameters:
    - A: 3D vector representing the rotation axis.
    - angle: Angle of rotation in radians.

    Returns:
    - R: 3x3 rotation matrix.
    """
    A = A / np.linalg.norm(A)
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.zeros((3, 3))
    R[0, :] = [A[0]**2 + (1 - A[0]**2) * c, A[0] * A[1] * (1 - c) - A[2] * s, A[0] * A[2] * (1 - c) + A[1] * s]
    R[1, :] = [A[0] * A[1] * (1 - c) + A[2] * s, A[1]**2 + (1 - A[1]**2) * c, A[1] * A[2] * (1 - c) - A[0] * s]
    R[2, :] = [A[0] * A[2] * (1 - c) - A[1] * s, A[1] * A[2] * (1 - c) + A[0] * s, A[2]**2 + (1 - A[2]**2) * c]
    return R


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def calculate_knnDist(df_check:pd.DataFrame):
    for index, row in df_check.iterrows():
        current_point = np.array([row['start_x'], row['start_y'], row['start_z']])
        mask_otherBr = df_check['branchID'] != row['branchID']
        rows_otherBr = df_check[mask_otherBr].copy()

        # Initialize the kNN model
        n_neighbors = 5
        knn_model = NearestNeighbors(n_neighbors=n_neighbors)

        # Fit the model to the features
        knn_model.fit(rows_otherBr[['start_x', 'start_y', 'start_z']].values)

        # Find the indices of the k nearest neighbors
        knnDists, _ = knn_model.kneighbors([current_point])

        # Calculate average distance
        avg_distance = knnDists.mean()
        df_check.loc[index, 'knnDist'] = avg_distance

    return df_check


def point_to_voxel(point, voxSize, voxDim):
    """
    Maps a 3D point to the voxel it belongs to.

    Args:
        point (tuple or list): A 3D point represented as (x, y, z).
        voxSize (float): The length of each voxel.

    Returns:
        tuple: The voxel position (i, j, k) corresponding to the input point.
    """
    # Calculate the voxel indices along each axis
    i = int((point[0] + voxSize * voxDim[0] / 2) // voxSize)
    j = int((point[1] + voxSize * voxDim[1] / 2) // voxSize)
    k = int(point[2] // voxSize)

    return i, j, k


def voxpos_to_corner(voxPos, voxSize, voxDim):
    """
    Maps a 3D voxel position to the its least corner coordinate.

    Args:
        coxPos (tuple or list): A 3D point represented as (x, y, z).
        voxSize (float): The length of each voxel.

    Returns:
        tuple: The voxel coner coordinate (x, y, z) corresponding to the input voxel position.
    """
    # Calculate the voxel indices along each axis
    x = voxSize * (voxPos[0] - voxDim[0] / 2)
    y = voxSize * (voxPos[1] - voxDim[1] / 2)
    z = voxSize * voxPos[2]

    return (x, y, z)


def plot_LAD (preLAD, fileName: str):
    # plot the LAD in voxels
    # Create a 3D figure
    fig = plt.figure(figsize=(12, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Set your threshold
    threshold = 0.05

    # Create a mask based on the threshold
    mask = preLAD > threshold

    # Normalize preLAD values to between 0.5 and 1 for transparency
    alpha = (preLAD - threshold) / (preLAD.max() - threshold) * 0.5 + 0.5
    #alpha = np.clip(alpha, 0.5, 1)

    # Create a color map, with alpha (transparency) based on normalized values
    colors = np.zeros((*alpha.shape, 4), dtype=object)
    #colors[mask] = [[0, 1, 0, alp] for alp in alpha[mask]]  # RGBA, green color, alpha based on normalized value
    colors[..., :3] = [0, 1, 0]  # Set the RGB channels
    colors[..., 3] = alpha  # Set the alpha channel

    # Voxel plot
    ax.set_aspect('equal')  # Aspect ratio is 1:1:1 in data space
    ax.voxels(mask, facecolors=colors, edgecolors="black", linewidth=0.2)

    #plt.show()
    plt.savefig(f'savefig_simp/{fileName}.png')


# ## 2.2 main function

def setInitialLAD (voxDim):
    voxData = np.array([[[0 for _ in range(voxDim[2])] for _ in range(voxDim[1])] for _ in range(voxDim[0])], dtype=np.float32)

    # create one seed point to grow the target crown
    seedNum = 1
    seeds = generateSeedsInit(voxDim, z_max=8)
    
    # initiate seeds in the voxData
    for start in seeds:
        startLAD = np.float32(np.random.uniform(0.05, 0.2))
        voxData[start[0]][start[1]][start[2]] = startLAD
        #print(f'seed position {start} is set with a targetLAD {startLAD}')
    
    # expansion the crown from the starts for N rounds
    n = 5
    threshold = 0.05
    maxAddStep = 0.08
    for _ in range(n):
        for start in np.argwhere(voxData > threshold):
            voxData = crownExpansion(voxData, start, maxAddStep)
    
    return voxData


def setTargetLAD (voxDim):
    voxData = np.array([[[0 for _ in range(voxDim[2])] for _ in range(voxDim[1])] for _ in range(voxDim[0])], dtype=np.float32)

    # create several seed points to grow the target crown
    seedNum = np.random.randint(1,5)
    seeds = generateSeedsTar(voxDim, seedNum, z_min=12)
    
    # initiate seeds in the voxData
    for start in seeds:
        startLAD = np.float32(np.random.uniform(0.05, 0.2))
        voxData[start[0]][start[1]][start[2]] = startLAD
        #print(f'seed position {start} is set with a targetLAD {startLAD}')
    
    # expansion the crown from the starts for N rounds
    n = 5
    threshold = 0.05
    maxAddStep = 0.08
    for _ in range(n):
        for start in np.argwhere(voxData > threshold):
            voxData = crownExpansion(voxData, start, maxAddStep)
    
    return voxData


def calculateScore_IoU (targetLAD, predictLAD, thresholdLAD):
    if targetLAD.shape != predictLAD.shape:
        raise ValueError("Both arrays must have the same shape.")
        
    if (np.isnan(targetLAD).any() or np.isnan(predictLAD).any()):
        return -1
        
    targetLAD[targetLAD < thresholdLAD] = 0
    predictLAD[predictLAD < thresholdLAD] = 0
    
    # Calculate the solid voxels
    solid_target = targetLAD > thresholdLAD
    solid_predict = predictLAD > thresholdLAD
    
    # Calculate the IoU score
    score = np.logical_and(solid_target,solid_predict).sum() / np.logical_or(solid_target, solid_predict).sum()
    #score = np.linalg.norm(targetLAD-predictLAD)
    
    return score


def initiatePrunGame(Ep):
    global projektNo
    projektNo = np.random.randint(10000)
    
    global voxSize
    voxSize = 0.8
    global voxDim
    voxDim = [20, 20, 26] # total voxel numbers along x, y and z axis
    
    # set a initial LAD values for each voxel
    global predictLAD
    predictLAD = setInitialLAD (voxDim)
    
    # set a target LAD values for each voxel
    global targetLAD
    targetLAD = setTargetLAD (voxDim)
    
    thresholdLAD = 0.05
    global score
    score = calculateScore_IoU(targetLAD, predictLAD, thresholdLAD)
    # changes end
    
    # Check if the folder already exists; if not, create it
    if not os.path.exists("savefig_simp"):
        os.makedirs("savefig_simp")
    #else:
        #print(f"Folder 'savefig_simp' already exists.")

    # plot current QSM model and voxel Model
    #fileNameLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd00_Sc{score:.2f}_LAD"   
    #plot_LAD(predictLAD, fileNameLAD)
    #fileNameTarLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd00_targetLAD"
    #plot_LAD(targetLAD, fileNameTarLAD)
    
    state_initiate = predictLAD - targetLAD # this is the first state of the episode
    state_initiate.astype(np.float32)

    return state_initiate

# # 3. Step
# ## 3.1 define basic functions

def voxExpansion (voxData, maxAddStep):
    # expansion the crown from the starts for N rounds
    n = 2
    threshold = 0.05
    for _ in range(n):
        for start in np.argwhere(voxData > threshold):
            voxData = crownExpansion(voxData, start, maxAddStep)
    return voxData


def voxThinning (voxData, std):
    mean = 0
    reducedLAD = abs(np.random.normal(mean, std, (voxData.shape[0], voxData.shape[1],voxData.shape[2])))
    return np.maximum(voxData - reducedLAD, 0)


def voxRaise (voxData, height):
    # Create a mask for values larger than 0.05
    mask = voxData > 0.05
    # Get the z-coordinates of the masked values
    z_coords = np.where(mask)[2]
    # Find the minimum z-coordinate
    min_z = np.min(z_coords)
    # Set the corresponding values to zero
    voxData[:,:,:(min_z+height)] = 0
    return voxData


def voxReductionTop (voxData, dist):
    # Create a mask for values larger than 0.05
    mask = voxData > 0.05
    # Get the z-coordinates of the masked values
    z_coords = np.where(mask)[2]
    # Find the maximum z-coordinate
    max_z = np.max(z_coords)
    # Set the corresponding values to zero
    voxData[:,:,(max_z-dist):] = 0
    return voxData


def voxReductionWest (voxData, dist):
    # Create a mask for values larger than 0.05
    mask = voxData > 0.05
    # Get the x-coordinates of the masked values
    x_coords = np.where(mask)[0]
    # Find the minimum x-coordinate
    min_x = np.min(x_coords)
    # Set the corresponding values to zero
    voxData[:(min_x+dist),:,:] = 0
    return voxData


def voxReductionEast (voxData, dist):
    # Create a mask for values larger than 0.05
    mask = voxData > 0.05
    # Get the x-coordinates of the masked values
    x_coords = np.where(mask)[0]
    # Find the maximum x-coordinate
    max_x = np.max(x_coords)
    # Set the corresponding values to zero
    voxData[(max_x-dist):,:,:] = 0
    return voxData


def voxReductionSouth (voxData, dist):
    # Create a mask for values larger than 0.05
    mask = voxData > 0.05
    # Get the y-coordinates of the masked values
    y_coords = np.where(mask)[1]
    # Find the minimum y-coordinate
    min_y = np.min(y_coords)
    # Set the corresponding values to zero
    voxData[:,:(min_y+dist),:] = 0
    return voxData


def voxReductionNorth (voxData, dist):
    # Create a mask for values larger than 0.05
    mask = voxData > 0.05
    # Get the y-coordinates of the masked values
    y_coords = np.where(mask)[1]
    # Find the maximum y-coordinate
    max_y = np.max(y_coords)
    # Set the corresponding values to zero
    voxData[:,(max_y-dist):,:] = 0
    return voxData


def distToCenter (voxData):
    # Collect voxel centers with values greater than 0.05
    voxelCenters = np.argwhere(voxData > 0.05)
    dists = np.zeros_like(voxData, dtype=np.float32)
    meanCenter = np.mean(voxelCenters,axis=0)
    center = np.array([meanCenter[0], meanCenter[1], meanCenter[2]])

    for i in range(voxData.shape[0]):
        for j in range(voxData.shape[1]):
            for k in range(voxData.shape[2]):
                voxel_center = np.array([i, j, k])
                dists[i, j, k] = np.linalg.norm(voxel_center - center)
                
    return dists


def voxShrink (voxData, rate):
    distToEdge = distToCenter (voxData)
    
    reducedLAD = np.power(distToEdge,2) * 0.001 * rate
    
    return np.maximum(voxData - reducedLAD, 0)


# ## 3.2 main function

def runPrunSimulation (action, episode, Rd, randTreeNo):
    try:
        done = False

        # updata dfQSM based on action input
        logger.info(f"++ Chosen action : [{action[0]},{action[1]}]")
        if action[0]==0: # THINNING
            LAD_updated = voxThinning(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==1: # RAISING
            LAD_updated = voxRaise(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==2: # REDUCTION_EAST
            LAD_updated = voxReductionEast(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==3: # REDUCTION_SOUTH
            LAD_updated = voxReductionSouth(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==4: # REDUCTION_WEST
            LAD_updated = voxReductionWest(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==5: # REDUCTION_NORTH
            LAD_updated = voxReductionNorth(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==6: # REDUCTION_TOP
            LAD_updated = voxReductionTop(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==7: # Topping
            LAD_updated = voxShrink(predictLAD, action[1])
            LAD_updated = voxExpansion(LAD_updated, 0.06)
        elif action[0]==8: # NOACTION
            LAD_updated = voxExpansion(predictLAD, 0.06)
        else:               # ENDING
            LAD_updated = predictLAD        
            done = True
        
        if Rd == 50:
            done = True
            
        reward_IoU = calculateScore_IoU (targetLAD, LAD_updated, 0.05)
        
        if Rd > 10:
            reward_IoU = reward_IoU - 0.002*(Rd-10) # if a high score can not be reached by 10 years, each extra year will cost 0.2 point reduction
        reward = reward_IoU
        #print("reward: ",reward)

        # plot current QSM model and voxel Model
        # the image name should follow "TreeXX_EpXX_Rd00_ScXXX_QSM/LAD"
        
        #fileNameLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd{Rd:02d}_Sc{reward:.2f}_LAD"
        #plot_LAD(LAD_updated, fileNameLAD)

        next_state = LAD_updated - targetLAD
        next_state.astype(np.float32)
        
        #print("next_state: ",next_state)

        return next_state, reward, done
       
    except: # it means the given parameter is not feasible
        return -targetLAD, -1, True #False
    
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
        
    def step(self, action):
        next_state, reward, done = runPrunSimulation(action, self.episodes, self.episode_steps, self.randTreeNo)
        #next_state, reward, done = runPrunSimulation(action)
        if next_state is not None:
            next_state = next_state #next_state.reshape(-1)
        # debug
        else:
            next_state = self.observation_space.sample() #.reshape(-1)
            
        return next_state, reward, done, {}
    
    # current state
    def reset(self):
        #state, randTreeNo = initiatePrunGame(self.episodes, self.episode_steps)
        state = initiatePrunGame(self.episodes)
        #self.randTreeNo = randTreeNo
        if state is not None:
            state = state #.reshape(-1)
        return state