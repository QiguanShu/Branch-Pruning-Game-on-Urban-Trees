#!/usr/bin/env python
# coding: utf-8

# # 1. Import

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
pd.options.mode.chained_assignment = None 

import joblib
import os

from sklearn.neighbors import NearestNeighbors # used for the function calculate_knnDist(df_check)
import itertools # for generating vosex spaces in the function calculateLAD(dfQSM)
import gym
from gym import spaces

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
    return voxData



# def generateSeeds(voxDim, seedNum, z_min):
#     all_locations = np.array([(x, y, z) for x in range(voxDim[0]) for y in range(voxDim[1]) for z in range(z_min, voxDim[2])])
#     indices = np.random.choice(all_locations.shape[0], seedNum, replace=False)
#     return all_locations[indices]

########################################## Updated 8th May 2024 ##########################################

def generateSeeds(voxDim, seedNum, z_min):
    margin = 5
    all_locations = np.array([(x, y, z) for x in range(margin, voxDim[0]-margin) for y in range(margin, voxDim[1]-margin) for z in range(z_min, voxDim[2]-margin)])
    indices = np.random.choice(all_locations.shape[0], seedNum, replace=False)
    return all_locations[indices]

########################################## Updated 8th May 2024 ##########################################

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



def plot_cylinders(cylinders, fig=None, nf=10, alp=1, Ind=None, fn="unnamed"):
  
    """
    Plots cylinders in 3D space.

    Parameters:
    - cylinders: List of dictionaries, each containing cylinder info (radius, length, start, axis, BranchOrder).
    - fig: Figure number (optional).
    - nf: Number of facets in the cylinders (optional, defaults to 10).
    - alp: Alpha value for transparency (optional, defaults to 0.5).
    - Ind: Indexes of cylinders to be plotted (optional, if not given, all cylinders are plotted).

    Example usage:
    cylinders = [
        {"radius": 0.1, "length": 0.2, "start": [0.2, 0.2, 0.1], "axis": [0.5, 0.5, 1.0], "BranchOrder": 0},
        # Add more cylinders here...
    ]
    plot_cylinders(cylinders)
    """    

    if fig is None:
        fig = plt.figure(figsize=(12, 20))
    ax = fig.add_subplot(111, projection='3d')

    for i, cylinder in enumerate(cylinders):
        Rad = cylinder["radius"]
        Len = cylinder["length"]
        Sta = np.array(cylinder["start"])
        Axe = np.array(cylinder["axis"])

        # Generate cylinder data
        theta = np.linspace(0, 2 * np.pi, nf+1)
        z = [0.00, 1.00]
        theta, z = np.meshgrid(theta, z)  # Create a grid
        x = np.cos(theta)
        y = np.sin(theta)

        # Scale
        x *= Rad
        y *= Rad
        z *= Len

        # Rotate
        ang = np.arccos(Axe[2])
        Axis = np.cross(np.array([0, 0, 1]).T, Axe.T)
        Rot = rotation_matrix(Axis, ang)
        C = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
        C = np.dot(C, Rot.T)

        # Translate
        C += Sta

        # Set the limits of x, y, and z axes
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        ax.set_zlim([0, 25])

        # Plot the cylinder
        ax.set_aspect('equal')  # Aspect ratio is 1:1:1 in data space
        ax.plot_surface(C[:, 0].reshape(2,nf+1), C[:, 1].reshape(2,nf+1), C[:, 2].reshape(2,nf+1), alpha=alp)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.savefig(f'savefig/{fn}.png')


def plot_QSM(df_check:pd.DataFrame, fileName:str):
    # transform the dataframe into a cylinder dictonray list
    cylinders = []
    for _, row in df_check.iterrows():
        cylinder = {"start":[row["start_x"],row["start_y"],row["start_z"]], "axis":[row["axis_x"],row["axis_y"],row["axis_z"]], "radius":row["radius"]+row["addRadius"], "length":row["length"]}
        cylinders.append(cylinder)
    plot_cylinders(cylinders, fn=fileName)


def reviseCylRadius (df_check:pd.DataFrame):
    # Sort DataFrame by parentCyID
    df_check = df_check.sort_values(by=['parentCyID'])
    df_check = df_check.reset_index(drop=True)

    # Filter parent rows (where 'parentCyID' is not null)
    parentIndex = df_check['parentCyID'].values

    for index, row in df_check.iterrows():
        # revise if the radius is larger than its parent
        if (row["radius"] > df_check.loc[df_check["cylinderID"] == parentIndex[index],"radius"]).any():
            df_check.loc[index, "radius"] = (df_check.loc[df_check["cylinderID"] == parentIndex[index],"radius"]).item()

    # Sort DataFrame by cylinderID
    df_check = df_check.sort_values(by=['cylinderID'])
    df_check = df_check.reset_index(drop=True)
    
    return df_check


def reviseCylAddRadius (df_check:pd.DataFrame):    
    # Sort DataFrame by parentCyID
    df_check = df_check.sort_values(by=['parentCyID'])
    df_check = df_check.reset_index(drop=True)

    # Filter parent rows (where 'parentCyID' is not null)
    parentIndex = df_check['parentCyID'].values

    for index, row in df_check.iterrows():
        # revise if the total radius is larger than its parent
        isParent = (df_check["cylinderID"] == parentIndex[index])
        if (df_check.loc[isParent,"branchID"] == row['branchID']).any:
            diff = (df_check.loc[isParent,"radius"]+df_check.loc[isParent,"addRadius"]) - (row["radius"]+row["addRadius"])
            if (diff > 0.02).any():
                df_check.loc[index, "addRadius"] = (df_check.loc[isParent,"addRadius"]).item()+(df_check.loc[isParent,"radius"]).item()-row["radius"] - 0.01
            elif (diff > 0.01).any():
                df_check.loc[index, "addRadius"] = (df_check.loc[isParent,"addRadius"]).item()+(df_check.loc[isParent,"radius"]).item()-row["radius"] - 0.05
            elif (diff > 0.005).any():
                df_check.loc[index, "addRadius"] = (df_check.loc[isParent,"addRadius"]).item()+(df_check.loc[isParent,"radius"]).item()-row["radius"] - 0.002

    # Sort DataFrame by cylinderID
    df_check = df_check.sort_values(by=['cylinderID'])
    df_check = df_check.reset_index(drop=True)
    
    return df_check


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


def loadModelLAD ():    
    # load the scaler
    sc=joblib.load('./pruneGameDependency/std_scaler_preLAD_3MainIndex.bin')
    # Load the model
    model_filename = './pruneGameDependency/HGBRegression_preLAD_3MainIndex.joblib'
    loaded_model = joblib.load(model_filename)
    
    return sc, loaded_model


def loadModelShoot ():    
    # load the scaler
    sc=joblib.load('./pruneGameDependency/std_scaler_PrunML_LGBM_forPrunGame.bin')
    # Load the model
    model_filename = './pruneGameDependency/PrunML_LBGM_forPrunGame.joblib'
    loaded_model = joblib.load(model_filename)

    return sc, loaded_model


def calculateQSMIndex (dfQSM:pd.DataFrame, voxSize, voxDim):
    # Initialize a 3D data structure for voxels (a nested list)
    # Each voxel will store (axis, branchID) information
    voxData = [[[None for _ in range(voxDim[2])] for _ in range(voxDim[1])] for _ in range(voxDim[0])]

    # Iterate through dfQSM rows to save retrived cylinder data in voxData
    for index, row in dfQSM.iterrows():
        midPoint = np.array([row["mid_x"], row["mid_y"], row["mid_z"]])

        # Map coordinates to voxel indices
        x, y, z = point_to_voxel(midPoint, voxSize, voxDim)

        # go through all the voxels within 2-voxel distance range to the center voxel
        for (i, j, k) in list(itertools.product(range(x-2,x+3), range(y-2,y+3), range(z-2,z+3))):
            # check if this voxel is within 1-voxel distance range, if so, add both axis and branchID dictionary
            if (i, j, k) in list(itertools.product(range(x-1,x+2), range(y-1,y+2), range(z-1,z+2))):
                # Check if the voxel already contains information
                if voxData[i][j][k]:
                    # add the dictionary item to the list
                    voxData[i][j][k].append({"axisZ": row["axis_z"], "branchID": row["branchID"]})
                else:
                    # Create a new list containing the first dictionary item
                    voxData[i][j][k] = [{"axisZ": row["axis_z"], "branchID": row["branchID"]}]
            # if out of the 1-voxel distance range, only add the branchID dictionary
            else:
                # Check if the voxel already contains information
                if voxData[i][j][k]:
                    # add the dictionary item to the list
                    voxData[i][j][k].append({"branchID": row["branchID"]})
                else:
                    # Create a new list containing the first dictionary item
                    voxData[i][j][k] = [{"branchID": row["branchID"]}]

    # calculate QSMIndex using the retrived voxData
    qsmIndex = [[[None for _ in range(voxDim[2])] for _ in range(voxDim[1])] for _ in range(voxDim[0])]

    for (i, j, k) in list(itertools.product(range(voxDim[0]), range(voxDim[1]), range(voxDim[2]))):

        if voxData[i][j][k]:
            # initiate the qsmIndex numbers
            cylNum2 = 0
            dire_z_sta2 = 0.0
            brIds = set()
            brLen_ave3 = 0

            # summary the data from voxData
            for data in voxData[i][j][k]:
                if "axisZ" in data:
                    cylNum2 += 1
                    dire_z_sta2 += data["axisZ"]
                if "branchID" in data:
                     brIds.add(data["branchID"])

            # finallize the qsmIndex calculation
            if cylNum2 > 0:
                dire_z_sta2 = dire_z_sta2 / cylNum2  
            brLenList = dfQSM[dfQSM['branchID'].isin(brIds)].groupby('branchID')['length'].sum().tolist()
            if brIds:  
                brLen_ave3 = sum(brLenList) / len(brLenList)

            qsmIndex[i][j][k] = [cylNum2, dire_z_sta2, brLen_ave3]

        else:
            qsmIndex[i][j][k] = [0, 0, 0]
    return qsmIndex


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

    # Create a color map, with alpha (transparency) based on normalized values
    colors = np.zeros((*alpha.shape, 4), dtype=object)
    colors[..., :3] = [0, 1, 0]  # Set the RGB channels
    colors[..., 3] = alpha  # Set the alpha channel

    # Voxel plot
    ax.set_aspect('equal')  # Aspect ratio is 1:1:1 in data space
    ax.voxels(mask, facecolors=colors, edgecolors="black", linewidth=0.2)

    #plt.show()
    plt.savefig(f'savefig/{fileName}.png')


# ## 2.2 main function
def setTargetLAD (voxDim):
    voxData = np.array([[[0 for _ in range(voxDim[2])] for _ in range(voxDim[1])] for _ in range(voxDim[0])], dtype=np.float32)

    # create several seed points to grow the target crown
    seedNum = np.random.randint(1,9)
    seeds = generateSeeds(voxDim, seedNum, z_min=8)
    
    # initiate seeds in the voxData
    for start in seeds:
        startLAD = np.float32(np.random.uniform(0.05, 0.2))
        voxData[start[0]][start[1]][start[2]] = startLAD
    
    # expansion the crown from the starts for N rounds
    n = 5
    threshold = 0.05
    maxAddStep = 0.08
    for _ in range(n):
        for start in np.argwhere(voxData > threshold):
            voxData = crownExpansion(voxData, start, maxAddStep)
    
    return voxData


def modifyQSM (df_check:pd.DataFrame):
    # move all the z-coordinates larger than 0
    min_start_z = df_check['start_z'].min() # Calculate the minimum value of 'start_z'
    df_check['start_z'] = df_check['start_z'] - min_start_z # Add the minimum value to all rows in 'start_z'
    
    # check the cylinder radius no larger than its parent
    df_check = reviseCylRadius(df_check)
    
    # filter the QSM by deleting branches with only one cylinder
    mask = df_check.groupby('branchID')['branchID'].transform('size') > 1 # Create a boolean mask for rows with duplicate branchID
    df_check = df_check[mask]
    
    # create new QSM columes for storing additional information
    df_check['knnDist'] = pd.Series(dtype='float32')   # the average distance to the k nearest cylinders
    df_check['addRadius'] = pd.Series(dtype='float32') # total radius increment on the cylinder
    df_check = calculate_knnDist(df_check)
    df_check['addRadius'] = 0
    # the middle point of eych cylinder will be used to assign cylinders to voxels
    df_check['mid_x'], df_check['mid_y'], df_check['mid_z'] = (
    df_check['start_x'] + df_check['axis_x'] * df_check['length'] / 2,
    df_check['start_y'] + df_check['axis_y'] * df_check['length'] / 2,
    df_check['start_z'] + df_check['axis_z'] * df_check['length'] / 2 )
    # adding columns for growth prediction
    df_check['isCutCylinder'] = False
    df_check['isPrunPos'] = False
    df_check['isShootPos'] = False
    
    return df_check


def calculateLAD (dfQSM:pd.DataFrame, voxSize, voxDim, scaler, trainedModel):
    # retrive QSM indexes for voxels (a nested list)
    qsmIndex = calculateQSMIndex (dfQSM, voxSize, voxDim)
    
    # predict LAD with the tranined model
    qsmIndex_flat = np.array(qsmIndex).reshape(-1, 3)  # Flatten the list
    x_scaled = scaler.transform(qsmIndex_flat)  # Scale the input data using the loaded scaler
    y_pred = trainedModel.predict(x_scaled)  # Predict y using the loaded model
    preLAD = y_pred.reshape(np.array(qsmIndex).shape[:-1]) # Reshape the predictions to match the original 3D list structure
    preLAD.astype(np.float32)

    return preLAD

''' unused function after updates
def calculateScore (targetLAD, predictLAD, rangeLAD):
    if targetLAD.shape != predictLAD.shape:
        raise ValueError("Both arrays must have the same shape.")
    
    # Calculate the Euclidean distance
    distance = np.sqrt(np.sum((targetLAD - predictLAD) ** 2))
    
    # Calculate the maximum possible distance
    max_distance = np.sqrt(np.sum((np.full(targetLAD.shape, rangeLAD[1]) - np.full(targetLAD.shape, rangeLAD[0])) ** 2))
    
    # Calculate the score
    score = 100 * (1 - distance / max_distance)
    
    return score
'''

## This is the init
def initiatePrunGame(episode, Rd):
    # load a random QSM model of the 7 given young plane tree examples
    global randTreeNo
    randTreeNo = np.random.randint(1, 8)
    df_source = pd.read_csv('./pruneGameDependency/OptQSM_R1016_P1to7.csv')
    df = df_source[df_source['treeID'] == randTreeNo]
    df = df.drop(columns=['fieldID', 'treeID', 'addedVirtual'])
    
    # modify the QSM data for the game
    global dfQSM
    dfQSM = modifyQSM(df)
    
    # set a target LAD values for each voxel
    global voxSize
    voxSize = 0.8
    global voxDim
    voxDim = [20, 20, 26] # total voxel numbers along x, y and z axis
    global targetLAD
    targetLAD = setTargetLAD (voxDim)
    
    # predict current LAD
    global scalerLAD
    global trainedModelLAD
    scalerLAD, trainedModelLAD = loadModelLAD()
    predictLAD = calculateLAD (dfQSM, voxSize, voxDim, scalerLAD, trainedModelLAD)
#     global scalerShoot
#     global trainedModelShoot
#     scalerShoot, trainedModelShoot = loadModelShoot() # for later use in simulating tree growth in each step
    
    # calculate the score
    #rangeLAD = [0, 0.2]
    thresholdLAD = 0.05
    global score
    score = calculateScore_IoU(targetLAD, predictLAD, thresholdLAD)
    
    # Check if the folder already exists; if not, create it
    if not os.path.exists("savefig"):
        os.makedirs("savefig")
    #else:
        #print(f"Folder 'savefig' already exists.")

    # the image name should follow "TreeXX_EpXX_Rd00_ScXXX_QSM/LAD"
    # The name means: tree number/ current episode number / current round in this episode / Score (in the iniating stage, the round is 00)
    if episode%5 == 0: # plot only every 5 epispodes
        fileNameQSM = "Tree{}_Ep{}_Rd{}_Sc{:.2f}_QSM".format(randTreeNo, episode, Rd, score)
        plot_QSM(dfQSM, fileNameQSM)
        #fileNameLAD = "TreeXX_EpXX_Rd00_ScXXX_LAD"
        fileNameLAD = "Tree{}_Ep{}_Rd{}_Sc{:.2f}_LAD".format(randTreeNo, episode, Rd, score)

        plot_LAD(predictLAD, fileNameLAD)
        #fileNameTarLAD = "TreeXX_EpXX_Rd00_ScXXX_targetLAD"
        fileNameTarLAD = "Tree{}_Ep{}_Rd{}_Sc{}_targetLAD".format(randTreeNo, episode, Rd, score)

        plot_LAD(targetLAD, fileNameTarLAD)
    
    state_initiate = predictLAD - targetLAD
    state_initiate.astype(np.float32)

    return state_initiate, randTreeNo
    
# # 3. Step

# ## 3.1 define basic functions

# pass the cut branch to all its children 
def passToChildren (df_check:pd.DataFrame, colName: str):
    
    # Sort DataFrame by parentCyID
    df_check = df_check.sort_values(by=['parentCyID'])
    df_check = df_check.reset_index(drop=True)
    parentIndex = df_check['parentCyID'].values

    for index, row in df_check.iterrows():
        if (row[colName] == False and (df_check.loc[df_check["cylinderID"] == parentIndex[index], colName] == True).any()):
            #print (f'cylinder {row["cylinderID"]} with index {index} and parent cylinder {parentIndex[index]}, rewrite {colName} from {row[colName]} to {(df_check.loc[df_check["cylinderID"] == parentIndex[index], colName]).item()}')
            df_check.loc[index,colName] = True

    # Sort DataFrame by cylinderID
    df_check = df_check.sort_values(by=['cylinderID'])
    df_check = df_check.reset_index(drop=True)
    
    return df_check


# define defferent pruning strategies
def thinningByDist (df_check:pd.DataFrame, minDist: np.float32):        
    for index, row in df_check.iterrows():
        if row['knnDist'] < minDist:
            df_check.loc[index, 'isPrunPos'] = True
    
    df_check['isCutCylinder'] = df_check['isPrunPos']
    df_check = passToChildren(df_check, 'isCutCylinder')
    return df_check

def raisingByHeight (df_check:pd.DataFrame, raiseH: np.float32):        
    nowH = df_check.loc[df_check['branchOrder'] > 1, 'start_z'].min()
    minH = nowH + raiseH
    for index, row in df_check.iterrows():
        if row['branchOrder'] > 0 and row['start_z'] < minH:
            df_check.loc[index,'isPrunPos'] = True
    
    df_check['isCutCylinder'] = df_check['isPrunPos']
    df_check = passToChildren(df_check, 'isCutCylinder')
    return df_check

def reductionByDirect (df_check:pd.DataFrame, fromDire: str, depth: np.float32):        
    if fromDire == 'south':
        min_y = df_check['start_y'].min()
        reduCyl = df_check[(df_check['branchOrder'] > 0) & (df_check['start_y'] >= min_y) & (df_check['start_y'] <= min_y + depth)]
    elif fromDire == 'north':
        max_y = df_check['start_y'].max()
        reduCyl = df_check[(df_check['branchOrder'] > 0) & (df_check['start_y'] <= max_y) & (df_check['start_y'] >= max_y - depth)]
    elif fromDire == 'west':
        min_x = df_check['start_x'].min()
        reduCyl = df_check[(df_check['branchOrder'] > 0) & (df_check['start_x'] >= min_x) & (df_check['start_x'] <= min_x + depth)]
    elif fromDire == 'east':
        max_x = df_check['start_x'].max()
        reduCyl = df_check[(df_check['branchOrder'] > 0) & (df_check['start_x'] <= max_x) & (df_check['start_x'] >= max_x - depth)]
    elif fromDire == 'up':
        max_z = df_check['start_z'].max()
        reduCyl = df_check[(df_check['start_z'] <= max_z) & (df_check['start_z'] >= max_z - depth)]
    else:
        print("Invalid direction. Please choose from 'south', 'north', 'west', 'east', 'up'.")
    
    df_check['isPrunPos'] = df_check.index.isin(reduCyl.index)        
    df_check['isCutCylinder'] = df_check['isPrunPos']
    df_check = passToChildren(df_check, 'isCutCylinder')
    return df_check

def toppingByCyNum (df_check:pd.DataFrame, cyNum: int):        
    df_brID = df_check.groupby('branchID')
    topCyl = df_brID.apply(lambda x: x.nlargest(cyNum, 'posInBranch') if len(x) >= cyNum else x)
    
    df_check['isPrunPos'] = df_check.index.isin([x[1] for x in topCyl.index])            
    df_check['isCutCylinder'] = df_check['isPrunPos']
    df_check = passToChildren(df_check, 'isCutCylinder')
    return df_check


def calculate_pitch_yaw(unitVec):
    # Ensure the vector is a unit vector
    assert np.isclose(np.sqrt(unitVec[0]**2 + unitVec[1]**2 + unitVec[2]**2), 1, rtol=1e-5), "The input vector must be a unit vector"

    # Calculate pitch (the angle with positive z axis in degrees)
    theta = np.degrees(np.arccos(unitVec[2]))

    # Calculate yaw (the angle with positive x axis in degrees)
    phi = np.degrees(np.arctan2(unitVec[1], unitVec[0]))

    return theta, phi


def angles_to_unit_vector(theta, phi):
    # Convert angles from degrees to radians
    theta = np.radians(theta)
    phi = np.radians(phi)

    # Calculate the components of the unit vector
    x0 = np.cos(phi) * np.sin(theta)
    y0 = np.sin(phi) * np.sin(theta)
    z0 = np.cos(theta)
    unitVec = [x0, y0, z0]

    return unitVec


def randomRule(lowBound, upBound, refNum, minDist, maxDist, max_attempts=100):
    flag=False
    loopCount=0
    
    while flag!=True:
        for _ in range(max_attempts):
            # Generate a random number in the range (lower Boundary, Upper Boundary)
            num = np.float32(np.random.uniform(lowBound, upBound))

            # If the difference to refNum is within the range (minDist, maxDist), regenerate the number
            dist = abs(num-refNum)
            if (dist > minDist) & (dist < maxDist) :
                return num
            
        # make the range softer that the previous 100 attempts
        minDist = minDist*0.8 
        maxDist = maxDist*1.2
        
        if maxDist > upBound:
            flag=True
    
        loopCount += 1
        if loopCount > 20: # limit the maximum expansion turns to be 20
            flag=True
    
    # If no suitable number is not found after max_attempts, return a random number in the range (lower Boundary, Upper Boundary)
    return np.float32(np.random.uniform(lowBound, upBound))



def generateShoot (df_check:pd.DataFrame):
    newCyl = pd.DataFrame()
    newCyl['branchID'] = pd.Series(dtype='int16')
    newCyl['branchOrder'] = pd.Series(dtype='int16')
    newCyl['cylinderID'] = pd.Series(dtype='int16')
    newCyl['posInBranch'] = pd.Series(dtype='int16')
    newCyl['parentCyID'] = pd.Series(dtype='int16')
    newCyl['childCyID'] = pd.Series(dtype='int16')
    newCyl['start_x'] = pd.Series(dtype='float32')
    newCyl['start_y'] = pd.Series(dtype='float32')
    newCyl['start_z'] = pd.Series(dtype='float32')
    newCyl['axis_x'] = pd.Series(dtype='float32')
    newCyl['axis_y'] = pd.Series(dtype='float32')
    newCyl['axis_z'] = pd.Series(dtype='float32')
    newCyl['length'] = pd.Series(dtype='float32')
    newCyl['radius'] = pd.Series(dtype='float32')
    newCyl['knnDist'] = pd.Series(dtype='float32')
    newCyl['addRadius'] = pd.Series(dtype='float32')
    newCyl['mid_x'] = pd.Series(dtype='float32')
    newCyl['mid_y'] = pd.Series(dtype='float32')
    newCyl['mid_z'] = pd.Series(dtype='float32')
    newCyl['isCutCylinder'] = pd.Series(dtype='boolean')
    newCyl['isPrunPos'] = pd.Series(dtype='boolean')
    newCyl['isShootPos'] = pd.Series(dtype='boolean')
    
    maxCyID = df_check['cylinderID'].max()
    maxBrID = df_check['branchID'].max()
    shootPos = df_check['isShootPos']
    countBr = 0
    shootCyNum = 5
    
    for index, row in df_check.iterrows():
        if shootPos[index]:
            countBr += 1
            for i in range(shootCyNum):
                ownIndex = shootCyNum*(countBr-1) + i
                if row['childCyID'] == 0:
                    newCyl.loc[ownIndex, "branchID"] = row['branchID']
                else:
                    newCyl.loc[ownIndex, "branchID"] = maxBrID + countBr
                if row['childCyID'] == 0:
                    newCyl.loc[ownIndex, "branchOrder"] = row['branchOrder']
                else:
                    newCyl.loc[ownIndex, "branchOrder"] = row['branchOrder'] + 1
                newCyl.loc[ownIndex, "cylinderID"] = maxCyID + 1 + ownIndex
                if row['childCyID'] == 0:
                    newCyl.loc[ownIndex, "posInBranch"] = row['posInBranch'] + i + 1
                else:
                    newCyl.loc[ownIndex, "posInBranch"] = i + 1
                if i == 0:
                    newCyl.loc[ownIndex, "parentCyID"] = row['cylinderID']
                else:
                    newCyl.loc[ownIndex, "parentCyID"] = maxCyID + ownIndex
                if i < shootCyNum - 1:
                    newCyl.loc[ownIndex, "childCyID"] = (maxCyID + 1 + ownIndex) + 1
                else:
                    newCyl.loc[ownIndex, "childCyID"] = 0
                if i == 0:
                    newCyl.loc[ownIndex, "start_x"] = row['start_x'] + row['axis_x'] * row['length']
                    newCyl.loc[ownIndex, "start_y"] = row['start_y'] + row['axis_y'] * row['length']
                    newCyl.loc[ownIndex, "start_z"] = row['start_z'] + row['axis_z'] * row['length']
                else:
                    newCyl.loc[ownIndex, "start_x"] = newCyl.loc[ownIndex-1, "start_x"] + newCyl.loc[ownIndex-1, "axis_x"] * newCyl.loc[ownIndex-1, "length"]
                    newCyl.loc[ownIndex, "start_y"] = newCyl.loc[ownIndex-1, "start_y"] + newCyl.loc[ownIndex-1, "axis_y"] * newCyl.loc[ownIndex-1, "length"]
                    newCyl.loc[ownIndex, "start_z"] = newCyl.loc[ownIndex-1, "start_z"] + newCyl.loc[ownIndex-1, "axis_z"] * newCyl.loc[ownIndex-1, "length"]
                if i == 0:
                    v_x = row['axis_x']
                    v_y = row['axis_y']
                    v_z = row['axis_z']
                    angZ, angX = calculate_pitch_yaw([v_x, v_y, v_z])
                    if row['childCyID'] == 0:
                        newAngZ = randomRule(0,100,angZ,0,5)
                        newAngX = randomRule(-180,180,angX,0,5)
                    else:
                        newAngZ = randomRule(0,100,angZ,30,60)
                        newAngX = randomRule(-180,180,angX,0,60)
                else:
                    v_x = newCyl.loc[ownIndex-1, "axis_x"]
                    v_y = newCyl.loc[ownIndex-1, "axis_y"]
                    v_z = newCyl.loc[ownIndex-1, "axis_z"]
                    angZ, angX = calculate_pitch_yaw([v_x, v_y, v_z])
                    newAngZ = randomRule(0,120,angZ,0,5)
                    newAngX = randomRule(-180,180,angX,0,5)
                newAxis = angles_to_unit_vector(newAngZ, newAngX)
                newCyl.loc[ownIndex, "axis_x"] = newAxis[0]
                newCyl.loc[ownIndex, "axis_y"] = newAxis[1]
                newCyl.loc[ownIndex, "axis_z"] = newAxis[2]
                newCyl.loc[ownIndex, "length"] = np.random.uniform(0.05, 0.1)*(np.cos(np.deg2rad(newAngZ))+1)
                newCyl.loc[ownIndex, "radius"] = np.random.uniform(0.005, 0.01)
                newCyl.loc[ownIndex, "knnDist"] = 0
                newCyl.loc[ownIndex, "addRadius"] = 0
                newCyl.loc[ownIndex, "mid_x"] = newCyl.loc[ownIndex, "start_x"] + newCyl.loc[ownIndex, "axis_x"] * newCyl.loc[ownIndex, "length"] / 2
                newCyl.loc[ownIndex, "mid_y"] = newCyl.loc[ownIndex, "start_y"] + newCyl.loc[ownIndex, "axis_y"] * newCyl.loc[ownIndex, "length"] / 2
                newCyl.loc[ownIndex, "mid_z"] = newCyl.loc[ownIndex, "start_z"] + newCyl.loc[ownIndex, "axis_z"] * newCyl.loc[ownIndex, "length"] / 2
                newCyl.loc[ownIndex, "isCutCylinder"] = False
                newCyl.loc[ownIndex, "isPrunPos"] = False
                newCyl.loc[ownIndex, "isShootPos"] = False

    newCyl = reviseCylRadius(newCyl)
    return newCyl


def branchExtend (df_check:pd.DataFrame, Iter:int):
    newCyl = pd.DataFrame()
    newCyl['branchID'] = pd.Series(dtype='int16')
    newCyl['branchOrder'] = pd.Series(dtype='int16')
    newCyl['cylinderID'] = pd.Series(dtype='int16')
    newCyl['posInBranch'] = pd.Series(dtype='int16')
    newCyl['parentCyID'] = pd.Series(dtype='int16')
    newCyl['childCyID'] = pd.Series(dtype='int16')
    newCyl['start_x'] = pd.Series(dtype='float32')
    newCyl['start_y'] = pd.Series(dtype='float32')
    newCyl['start_z'] = pd.Series(dtype='float32')
    newCyl['axis_x'] = pd.Series(dtype='float32')
    newCyl['axis_y'] = pd.Series(dtype='float32')
    newCyl['axis_z'] = pd.Series(dtype='float32')
    newCyl['length'] = pd.Series(dtype='float32')
    newCyl['radius'] = pd.Series(dtype='float32')
    newCyl['knnDist'] = pd.Series(dtype='float32')
    newCyl['addRadius'] = pd.Series(dtype='float32')
    newCyl['mid_x'] = pd.Series(dtype='float32')
    newCyl['mid_y'] = pd.Series(dtype='float32')
    newCyl['mid_z'] = pd.Series(dtype='float32')
    newCyl['isCutCylinder'] = pd.Series(dtype='boolean')
    newCyl['isPrunPos'] = pd.Series(dtype='boolean')
    newCyl['isShootPos'] = pd.Series(dtype='boolean')
    
    maxCyID = df_check['cylinderID'].max()
    maxBrID = df_check['branchID'].max()
    endPos = (df_check['childCyID']==0).values
    countBr = 0
    shootCyNum = 5
    
    for index, row in df_check.iterrows():
        if endPos[index]:
            countBr += 1
            df_check.loc[index, 'childCyID'] = maxCyID + 1 + shootCyNum*(countBr-1)
            for i in range(shootCyNum):
                ownIndex = shootCyNum*(countBr-1) + i
                newCyl.loc[ownIndex, "branchID"] = row['branchID']
                newCyl.loc[ownIndex, "branchOrder"] = row['branchOrder']
                newCyl.loc[ownIndex, "cylinderID"] = maxCyID + 1 + ownIndex
                newCyl.loc[ownIndex, "posInBranch"] = row['posInBranch'] + i + 1
                if i == 0:
                    newCyl.loc[ownIndex, "parentCyID"] = row['cylinderID']
                else:
                    newCyl.loc[ownIndex, "parentCyID"] = maxCyID + ownIndex
                if i < shootCyNum - 1:
                    newCyl.loc[ownIndex, "childCyID"] = (maxCyID + 1 + ownIndex) + 1
                else:
                    newCyl.loc[ownIndex, "childCyID"] = 0
                if i == 0:
                    newCyl.loc[ownIndex, "start_x"] = row['start_x'] + row['axis_x'] * row['length']
                    newCyl.loc[ownIndex, "start_y"] = row['start_y'] + row['axis_y'] * row['length']
                    newCyl.loc[ownIndex, "start_z"] = row['start_z'] + row['axis_z'] * row['length']
                else:
                    newCyl.loc[ownIndex, "start_x"] = newCyl.loc[ownIndex-1, "start_x"] + newCyl.loc[ownIndex-1, "axis_x"] * newCyl.loc[ownIndex-1, "length"]
                    newCyl.loc[ownIndex, "start_y"] = newCyl.loc[ownIndex-1, "start_y"] + newCyl.loc[ownIndex-1, "axis_y"] * newCyl.loc[ownIndex-1, "length"]
                    newCyl.loc[ownIndex, "start_z"] = newCyl.loc[ownIndex-1, "start_z"] + newCyl.loc[ownIndex-1, "axis_z"] * newCyl.loc[ownIndex-1, "length"]
                if i == 0:
                    v_x = row['axis_x']
                    v_y = row['axis_y']
                    v_z = row['axis_z']
                else:
                    v_x = newCyl.loc[ownIndex-1, "axis_x"]
                    v_y = newCyl.loc[ownIndex-1, "axis_y"]
                    v_z = newCyl.loc[ownIndex-1, "axis_z"]
                angZ, angX = calculate_pitch_yaw([v_x, v_y, v_z])
                newAngZ = randomRule(0,100,angZ,0,5)
                newAngX = randomRule(-180,180,angX,0,5)
                newAxis = angles_to_unit_vector(newAngZ, newAngX)
                newCyl.loc[ownIndex, "axis_x"] = newAxis[0]
                newCyl.loc[ownIndex, "axis_y"] = newAxis[1]
                newCyl.loc[ownIndex, "axis_z"] = newAxis[2]
                newCyl.loc[ownIndex, "length"] = np.random.uniform(0.05, 0.1)*(np.cos(np.deg2rad(newAngZ))+1)/(1+Iter/40) # the average length of extension would reduce as the tree gets older
                newCyl.loc[ownIndex, "radius"] = np.random.uniform(0.005, 0.01)
                newCyl.loc[ownIndex, "knnDist"] = 0
                newCyl.loc[ownIndex, "addRadius"] = 0
                newCyl.loc[ownIndex, "mid_x"] = newCyl.loc[ownIndex, "start_x"] + newCyl.loc[ownIndex, "axis_x"] * newCyl.loc[ownIndex, "length"] / 2
                newCyl.loc[ownIndex, "mid_y"] = newCyl.loc[ownIndex, "start_y"] + newCyl.loc[ownIndex, "axis_y"] * newCyl.loc[ownIndex, "length"] / 2
                newCyl.loc[ownIndex, "mid_z"] = newCyl.loc[ownIndex, "start_z"] + newCyl.loc[ownIndex, "axis_z"] * newCyl.loc[ownIndex, "length"] / 2
                newCyl.loc[ownIndex, "isCutCylinder"] = False
                newCyl.loc[ownIndex, "isPrunPos"] = False
                newCyl.loc[ownIndex, "isShootPos"] = False
    
    # combine new shoot cylinders to df_check
    df_check = pd.concat([df_check, newCyl])
    df_check = df_check.reset_index(drop=True)

    # update the radius, bud and knnDist of the df_check
    df_check = reviseCylRadius(df_check)
    df_check = reviseCylAddRadius(df_check)
    df_check = calculate_knnDist(df_check)
    
    return df_check



def preShootPosSimple (df_check:pd.DataFrame):
    # check if this position has sub branch already
    df_check['hasSubBr'] = False
    for index, row in df_check.iterrows():
        # Filter rows where 'parentID' matches the current row's 'cylinderID'
        matching_rows = df_check[df_check['parentCyID'] == row['cylinderID']]

        # Check if any of the matching rows have a different 'branchID'
        if matching_rows['branchID'].nunique() > 1:
            df_check.loc[index,'hasSubBr'] = True
    
    # new shoots are random with a minimum distances in between
    df_check = df_check.reset_index(drop=True)
    df_check = df_check.sort_values(['branchID','posInBranch'], ascending=[True, True])   # sort df by 'branchID' and 'posInBranch'
    shootPos = [False] * df_check.shape[0] #initiate a shoot pos list with no shoot
    minDist = 4
    distCount = 0
    lastBrID = -1
    for index, row in df_check.iterrows():
        if distCount > minDist:
            skipChance = 1/(row['branchOrder']/2+1.2)
            shootPos[index] = np.random.choice([True, False], 1,p=[1-skipChance,skipChance])[0]
        distCount+=1
        if shootPos[index]:
            distCount = 0
            continue
        if row['hasSubBr']:
            distCount = 0
            continue
        if row['branchID'] != lastBrID:
            distCount = 0
        lastBrID = row['branchID']
      
    return shootPos


def reviseChildCyID (df_check:pd.DataFrame):
    for index, row in df_check.iterrows():
        if row['isPrunPos']:
            mask_parentCyl = df_check['cylinderID'] == row['parentCyID']
            if (df_check.loc[mask_parentCyl, 'branchID'] == row['branchID']).any():
                df_check.loc[mask_parentCyl, 'childCyID'] = 0

    return df_check



# Define the logistic function
def logistic(x, L, k, x0):
    # L is the upper limit, k is the growth rate, and x0 is the x-value of the sigmoid’s midpoint
    return L / (1 + np.exp(-k*(x-x0)))


def calculateAddRadius (df_check:pd.DataFrame, Iter):
    for index, row in df_check.iterrows():
        # increament rate in relation to age is referenced to Dervishi et al. (2022) https://doi.org/10.3390/f13050641
        # DBH increment around 2cm at the beginning, 50 years around 0.7cm, 100+ years around 0.4cm
        increment = 0.2 / logistic(Iter, L=50, k=0.05, x0=30) / 2 / (row['branchOrder']/3 + row['posInBranch']/40 + 1)
        df_check.loc[index, 'addRadius'] += increment
    return df_check


def calculate_knnDistByRadius(df_check:pd.DataFrame):
    for index, row in df_check.iterrows():
        current_point = np.array([row['start_x'], row['start_y'], row['start_z']])
        mask_otherBr = (df_check['branchID'] != row['branchID']) &\
                        ((df_check['radius'] + df_check['addRadius']) > (row['radius'] + row['addRadius']) * 3/4) &\
                        (df_check['start_z'] >= row['start_z']-0.05) &\
                        (df_check['branchOrder'] >= row['branchOrder'])
        n_neighbors = 5
        if mask_otherBr.sum() >= n_neighbors:
            rows_otherBr = df_check[mask_otherBr].copy()

            # Initialize the kNN model
            knn_model = NearestNeighbors(n_neighbors=n_neighbors)

            # Fit the model to the features
            knn_model.fit(rows_otherBr[['start_x', 'start_y', 'start_z']].values)

            # Find the distances of the k nearest neighbors
            knnDists, _ = knn_model.kneighbors([current_point])
            avg_distance = knnDists.mean()
            df_check.loc[index, 'knnDist'] = avg_distance
        else:
            df_check.loc[index, 'knnDist'] = 99 # it means no enough cylinders from other branches have comparible sizes
    return df_check['knnDist']


# this is a simplified version of checkNaturalDeath(df_check), it will not repeately calculate knnDist again after removal of each single branch
def checkNaturalDeath_simple (df_check:pd.DataFrame):
    knnDistByRad = calculate_knnDistByRadius(df_check)
    for index, row in df_check.iterrows():
        minDist = 0.04
        minDistBr = (row['radius'] + row['addRadius']) * 5 
        saveChance = 1/(row['branchOrder']/4+1)
        if knnDistByRad[index] < minDist:            
            df_check.loc[index, 'isPrunPos'] = np.random.choice([True,False],1,p=[1-saveChance,saveChance])[0]
        elif knnDistByRad[index] < minDistBr:
            df_check.loc[index, 'isPrunPos'] = np.random.choice([True,False],1,p=[1-saveChance,saveChance])[0]

    df_check['isCutCylinder'] = df_check['isPrunPos']
    df_check = passToChildren(df_check, 'isCutCylinder')
    df_check = reviseChildCyID(df_check)

    #delete dead cylinders
    df_check = df_check[df_check['isCutCylinder']==False]
    df_check = df_check.reset_index(drop=True)
    
    df_check = reviseCylRadius(df_check)
    df_check = calculate_knnDist(df_check)

    return df_check



# ## 3.2 main function

#def completeQSMbyOperation (df_check:pd.DataFrame, scaler, trainedModel): # this line is the old version when using the trained shoot prediction model
def completeQSMbyOperation (df_check:pd.DataFrame):
    shootPos = preShootPosSimple(df_check)
    df_check = reviseChildCyID(df_check) # this step must be performed after the shoot prediction
    df_check['isShootPos'] = shootPos

    #delete cut cylinders
    df_check = df_check[df_check['isCutCylinder']==False]
    df_check = df_check.reset_index(drop=True)
  
    # combine new shoot cylinders to df_check
    newShootCyl = generateShoot(df_check)
    df_check = pd.concat([df_check, newShootCyl])
    df_check = df_check.reset_index(drop=True)

    # update the radius, bud and knnDist of the df_check
    df_check = reviseCylRadius(df_check)
    df_check = calculate_knnDist(df_check)
    
    return df_check


def updateQSMbyNaturalGrowth (df_check:pd.DataFrame, Rd):
    # erase cut position and branches
    df_check['isPrunPos'] = False
    df_check['isCutCylinder'] = False
    df_check['isShootPos'] = False
    # predict new shoots without operations
    shootPos = preShootPosSimple(df_check)
    df_check = reviseChildCyID(df_check)
    df_check['isShootPos'] = shootPos
    
    # combine new shoot cylinders to df_check
    newShootCyl = generateShoot(df_check)
    df_check = pd.concat([df_check, newShootCyl])
    df_check = df_check.reset_index(drop=True)
    
    # extend in branch length
    df_check = branchExtend(df_check, Rd)
    
    # increament in radius
    df_check = calculateAddRadius(df_check, Rd)
    
    # natural death
    df_check = checkNaturalDeath_simple(df_check) # a simpler version of the function checkNaturalDeath(df_check)

    return df_check

''' unused version after the update
def updateStateAndScore (df_check:pd.DataFrame, scaler, trainedModel, targetLAD):
    # calculate updated LAD
    preLAD_updated = calculateLAD (df_check, voxSize, voxDim, scaler, trainedModel)
    
    # calculate the score
    rangeLAD = [0, 0.2]
    score_updated = calculateScore(targetLAD, preLAD_updated, rangeLAD)
    return preLAD_updated, score_updated
'''

########################################## Updated 8th May 2024 ##########################################

def calculateScore_IoU (targetLAD, predictLAD, thresholdLAD):
    if targetLAD.shape != predictLAD.shape:
        raise ValueError("Both arrays must have the same shape.")
    
    # Calculate the solid voxels
    solid_target = targetLAD > thresholdLAD
    solid_predict = predictLAD > thresholdLAD
    
    # Calculate the IoU score
    score = np.logical_and(solid_target,solid_predict).sum() / np.logical_or(solid_target, solid_predict).sum() * 100
    
    return score

def updateStateAndScore_IoU (df_check:pd.DataFrame, scaler, trainedModel, targetLAD):
    # calculate updated LAD
    preLAD_updated = calculateLAD (df_check, voxSize, voxDim, scaler, trainedModel)
    preLAD_updated.astype(np.float32)
    
    # calculate the score
    thresholdLAD = 0.05
    score_updated = calculateScore_IoU(targetLAD, preLAD_updated, thresholdLAD)
    return preLAD_updated, score_updated

########################################## Updated 8th May 2024 ##########################################

## This is the step function
def runPrunSimulation (action, episode, Rd, randTreeNo):
    try:
        global dfQSM
        dfQSM['isPrunPos']=False # reset isPrunPos
        dfQSM['isCutCylinder']=False # reset isCutCylinder
        done = False

        # updata dfQSM based on action input
        if action[0]==0: # THINNING
            act_param = action[1]/10 * 0.05 # note
            dfQSM_updated = thinningByDist(dfQSM, act_param)
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        elif action[0]==1: # RAISING
            dfQSM_updated = raisingByHeight(dfQSM, action[1])
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        elif action[0]==2: # REDUCTION_EAST
            dfQSM_updated = reductionByDirect(dfQSM, 'east', action[1])
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        elif action[0]==3: # REDUCTION_SOUTH
            dfQSM_updated = reductionByDirect(dfQSM, 'south', action[1])
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        elif action[0]==4: # REDUCTION_WEST
            dfQSM_updated = reductionByDirect(dfQSM, 'west', action[1])
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        elif action[0]==5: # REDUCTION_NORTH
            dfQSM_updated = reductionByDirect(dfQSM, 'north', action[1])
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        elif action[0]==6: # REDUCTION_TOP
            # < 5
            act_param = action[1]/10 * 5
            dfQSM_updated = reductionByDirect(dfQSM, 'up', act_param)    
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        elif action[0]==7: # Topping
            # < 5
            act_param = action[1]/10 * 5
            dfQSM_updated = toppingByCyNum(dfQSM, act_param)
            dfQSM_updated = completeQSMbyOperation(dfQSM_updated)
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM_updated, Rd)
        #elif action[0]==8: # NOACTION
        else:
            dfQSM_updated = updateQSMbyNaturalGrowth(dfQSM, Rd)  
        # else:               # ENDING remove action
        #     dfQSM_updated = dfQSM           
        #     done = True
            
        ## Rd 30 done=true
        if Rd == 20:
            done = True
        
        ## 
        if dfQSM_updated.shape[0] > 20: # to ensure that the tree has something cylinders left after the operation
            # calculate new LAD and reward
            currentLAD, reward_IoU = updateStateAndScore_IoU (dfQSM_updated, scalerLAD, trainedModelLAD, targetLAD)
            dfQSM = dfQSM_updated # dfQSM will be passed to the next round
            if Rd > 10:
                reward_IoU = reward_IoU - 0.2*(Rd-10) # if a high score can not be reached by 10 years, each extra year will cost 0.2 point reduction
            reward = reward_IoU

            # plot current QSM model and voxel Model
            # the image name should follow "TreeXX_EpXX_Rd00_ScXXX_QSM/LAD"
            if episode%5 == 0: # plot only every 5 epispodes
                fileNameQSM = "Tree{}_Ep{}_Rd{}_Sc{:.2f}_QSM".format(randTreeNo, episode, Rd, reward)
                plot_QSM(dfQSM_updated, fileNameQSM)  
                #fileNameLAD = f"TreeXX_Ep{episode}_Rd{Rd}_ScXXX_LAD"
                fileNameLAD = "Tree{}_Ep{}_Rd{}_Sc{:.2f}_LAD".format(randTreeNo, episode, Rd, reward)
                plot_LAD(currentLAD, fileNameLAD)

            next_state = currentLAD - targetLAD
            next_state.astype(np.float32)

            return next_state, reward, done
        else:
            #return None, -1, False
            return -targetLAD, 0, True
       
    except: # it means the given parameter is not feasible
        #not terminiate the game, but only return to the previous round
        return -targetLAD, 0, True #None, -1, False
    
class PrunEnvWrapper(gym.Env):
    def __init__(self,):
        # Tree Number 
        self.randTreeNo = ""
        
        # State size, 1D vector flatten from 3D voxel
        self.dimension = 20*20*26
        self.observation_space = spaces.Box(0, 1, shape=(self.dimension,), dtype=np.float32)
        
        # Action space
        self.num_actions = 9 #10
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.num_actions), 
            spaces.Box(0, 10, shape=(1,), dtype=np.float32)
        ))
        
        self.episodes = 0
        self.episode_steps = 0 # Rd
        
    def step(self, action):
        next_state, reward, done = runPrunSimulation(action, self.episodes, self.episode_steps, self.randTreeNo)
        if next_state is not None:
            next_state = next_state.reshape(-1)
        # debug
        else:
            next_state = self.observation_space.sample().reshape(-1)
            
        return next_state, reward, done, {}
    
    # current state
    def reset(self):
        state, randTreeNo = initiatePrunGame(self.episodes, self.episode_steps)
        self.randTreeNo = randTreeNo
        if state is not None:
            state = state.reshape(-1)
        return state