#!/usr/bin/env python
# coding: utf-8

# # 1. Import
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
pd.options.mode.chained_assignment = None 

import os


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
    # new version add a margin restriction for the seed positions
    margin = 5
    all_locations = np.array([(x, y, z) for x in range(margin, voxDim[0]-margin) for y in range(margin, voxDim[1]-margin) for z in range(z_min, voxDim[2]-margin)])
    indices = np.random.choice(all_locations.shape[0], seedNum, replace=False)
    return all_locations[indices]


def generateSeedsInit(voxDim, z_max):
    # new version add a margin restriction for the seed positions
    margin = 5
    all_locations = np.array([(x, y, z) for x in range(margin, voxDim[0]-margin) for y in range(margin, voxDim[1]-margin) for z in range(margin, z_max)])
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
    plt.savefig(f'savefig/{fileName}.png')


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
    
    # Calculate the solid voxels
    solid_target = targetLAD > thresholdLAD
    solid_predict = predictLAD > thresholdLAD
    
    # Calculate the IoU score
    score = np.logical_and(solid_target,solid_predict).sum() / np.logical_or(solid_target, solid_predict).sum() * 100
    
    return score


def initiatePrunGame():
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
    
    # Check if the folder already exists; if not, create it
    if not os.path.exists("savefig"):
        os.makedirs("savefig")
    #else:
        #print(f"Folder 'savefig' already exists.")

    # plot current QSM model and voxel Model
    fileNameLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd00_Sc{score:.2f}_LAD"   
    plot_LAD(predictLAD, fileNameLAD)
    fileNameTarLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd00_targetLAD"
    plot_LAD(targetLAD, fileNameTarLAD)
    
    state_initiate = predictLAD - targetLAD # this is the first state of the episode
    state_initiate.astype(np.float32)


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

def runPrunSimulation (action):
    try:
        done = False

        # updata dfQSM based on action input
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
        
        
        reward_IoU = calculateScore_IoU (targetLAD, LAD_updated, 0.05)
        if Rd > 10:
            reward_IoU = reward_IoU - 0.2*(Rd-10) # if a high score can not be reached by 10 years, each extra year will cost 0.2 point reduction
        reward = reward_IoU

        # plot current QSM model and voxel Model
        # the image name should follow "TreeXX_EpXX_Rd00_ScXXX_QSM/LAD"
        fileNameLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd{Rd:02d}_Sc{reward:.2f}_LAD"
        plot_LAD(LAD_updated, fileNameLAD)

        next_state = LAD_updated - targetLAD
        next_state.astype(np.float32)

        return next_state, reward, done
       
    except: # it means the given parameter is not feasible
        return -targetLAD, -1, False
    

# # Game UI
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

import logging
from tkinter import messagebox
from datetime import datetime


class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        logging.Handler.__init__(self)
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.yview(tk.END)  # Autoscroll to the bottom


def save_log_to_file():
    log_messages = history_log.get("1.0", tk.END)
    
    # Open a file dialog for choosing the save location
    file_path = filedialog.asksaveasfilename(
        title="Save Game History",
        defaultextension=f"{datetime.now()}_player1_totalEp{Ep:04d}_Rd{Rd:02d}_finalSc{score:.2f}.txt",
        filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
    )

    if file_path:
        global logger
        try:
            with open(file_path, "w") as log_file:
                log_file.write(log_messages)
            logger.info(f"Log saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving log: {str(e)}")


def runPrunGame():
    global Rd
    global Ep
   
    # Create the main window
    root = tk.Tk()
    root.title("BranchPrunGame_V1.0")

    # Create a frame for the image boxes
    image_frame = tk.Frame(root)
    image_frame.pack(side=tk.TOP)
    
    # Create three image boxes
    text_labels = ["Current LAD State", "Target LAD State"]
    image_labels = []
    for i in range(2):
        label = tk.Label(image_frame, text=text_labels[i], bg="grey", width=80, height=40)
        label.pack(side=tk.LEFT, padx=5, pady=5)
        image_labels.append(label)
        

    # Create a frame for the console
    console_frame = tk.Frame(root)
    console_frame.pack(side=tk.TOP, fill=tk.X)

    # Create a dropdown list
    tk.Label(console_frame, text="Choose Operation:").pack(side=tk.LEFT)
    options = ['thinning',  
              'raising', 
              'reduction_east', 
              'reduction_south', 
              'reduction_west', 
              'reduction_north', 
              'reduction_up', 
              'topping', 
              'no_action', 
              'end_game']   
    global dropdown
    dropdown = ttk.Combobox(console_frame, values=options)
    dropdown.pack(side=tk.LEFT, padx=5)

    # Create an input box
    tk.Label(console_frame, text="Enter Parameter:").pack(side=tk.LEFT)
    global input_box
    input_box = tk.Entry(console_frame)
    input_box.pack(side=tk.LEFT, padx=5)

    def update_image():   
        # update the image
        fileNameLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd{Rd:02d}_Sc{score:.2f}_LAD"
        fileNameTarLAD = f"TreeTest{projektNo}_Ep{Ep:04d}_Rd00_targetLAD"
        file_paths = [f'savefig/{fileNameLAD}.png', f'savefig/{fileNameTarLAD}.png']
        for i, file_path in enumerate(file_paths):
            image = Image.open(file_path)
            photo = ImageTk.PhotoImage(image.resize((600,1000)))
            image_labels[i].config(image=photo, width=566, height=606, padx=5, pady=5)       
            image_labels[i].image = photo
            # Debugging: Print dimensions
            #print(f"Label size: {image_labels[i].winfo_width()} x {image_labels[i].winfo_height()}")
            #print(f"Photo size: {photo.width()} x {photo.height()}")
    
    
    # Create a button to restart the game
    def restart_game(): 
        # Initiate the Episode and Round
        global Ep
        global Rd
        Ep += 1
        Rd = 0
        initiatePrunGame()
        update_image()
        # update the label on the control pannel
        round_label.config(text=f"Round: {Rd:02d}")
        score_label.config(text=f"Score: {score:.2f}")

    restart_button = tk.Button(console_frame, text="Start/Restart", command=restart_game)
    restart_button.pack(side=tk.LEFT, padx=5)
    
    def next_round():
        global Ep
        global Rd
        Rd += 1
        actPara = np.float32(input_box.get())
        if dropdown.get() == "thinning":
            actOpt= 0
        elif dropdown.get() == "raising":
            actOpt= 1
            actPara = int(input_box.get())
        elif dropdown.get() == "reduction_east":
            actOpt= 2
            actPara = int(input_box.get())
        elif dropdown.get() == "reduction_south":
            actOpt= 3
            actPara = int(input_box.get())
        elif dropdown.get() == "reduction_west":
            actOpt= 4
            actPara = int(input_box.get())
        elif dropdown.get() == "reduction_north":
            actOpt= 5
            actPara = int(input_box.get())
        elif dropdown.get() == "reduction_up":
            actOpt= 6
            actPara = int(input_box.get())
        elif dropdown.get() == "topping":
            actOpt= 7
        elif dropdown.get() == "no_action":
            actOpt= 8
        else: #dropdown == "end_game"
            actOpt= 9
        myAction = (actOpt, actPara)
        global score
        global done
        next_state_tmp, score_tmp, done = runPrunSimulation(myAction) 
        if score_tmp != -1:
            next_state = next_state_tmp
            score = score_tmp
            # Log the simulation
            logger.info(f"Episode: {Ep:04d}; Round: {Rd:02d}; Action {myAction}; Score: {score:.2f}; Done: {done}")
            
            # update the images and the labels on the control pannel
            if not done:
                update_image()
            round_label.config(text=f"Round: {Rd:02d}")
            score_label.config(text=f"Score: {score:.2f}")

            # end the game
            if done == True:
                msg=messagebox.showinfo("Game Over",f"Episode {Ep} ends! Final Score: {score}! The game will restart!")
                save_log_to_file()
                restart_game()
        else:
            Rd -= 1
            # Log the faliure in the action
            logger.info(f"Episode: {Ep:04d}; Round: {Rd:02d}; Action {myAction} Failed! Please input another operation or parameter!")
        

    # Create a button to confirm input
    confirm_button = tk.Button(console_frame, text="Simulate Round", command=next_round)
    confirm_button.pack(side=tk.LEFT, padx=5)


    # Create labels to display the current round and score
    global Rd
    global score
    round_label = tk.Label(console_frame, text=f"Round: {Rd:02d}")
    round_label.pack(side=tk.LEFT, padx=5)
    score_label = tk.Label(console_frame, text=f"Score: {score:.2f}")
    score_label.pack(side=tk.LEFT, padx=5)

    # Create a frame for the history log
    history_frame = tk.Frame(root)
    history_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Create a scrollbar for the history log
    scrollbar = tk.Scrollbar(history_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create a Text widget for the history log
    global history_log
    history_log = tk.Text(history_frame, height=10, state='disabled')
    history_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Set up logging to display in the Text widget
    global logger
    logger = logging.getLogger("my_logger")
    text_handler = TextHandler(history_log)
    logger.addHandler(text_handler)
    logger.setLevel(logging.INFO)
    
    # Save Log button
    save_button = ttk.Button(history_frame, text="Save Log", command=save_log_to_file)
    save_button.pack(side=tk.RIGHT)

    root.mainloop()



global Rd
global Ep
Ep = 0
Rd = 0
score = 0
done = False
 
runPrunGame()