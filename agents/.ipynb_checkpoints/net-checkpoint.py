# @author Metro
# @time 2021/11/3

"""
  Ref: https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/net.py
       https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from logger import logger

epsilon = 1e-6

def init_(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DuelingDQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64),
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()
        
        # Convolutional layers to process state_dim with pooling layers
        self.conv1 = nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size of the output from the convolutional layers
        conv_output_size = 64 * (20 // 8) * (20 // 8) 
        state_dim = conv_output_size
        ############### OLD VERSION BELOW ################
        
        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim + action_dim, hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            
        self.adv_layers_1 = nn.Linear(hidden_layers[-1], action_dim)
        self.val_layers_1 = nn.Linear(hidden_layers[-1], 1)

        self.adv_layers_2 = nn.Linear(hidden_layers[-1], action_dim)
        self.val_layers_2 = nn.Linear(hidden_layers[-1], 1)

        self.apply(init_)

    def forward(self, state, action_params):
        
        # Pass the state through the convolutional and pooling layers
        state = torch.permute(state, (0,3,1,2))
        
        x = F.relu(self.bn1(self.conv1(state)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten the output from the convolutional layers
        #x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        
        temp = torch.cat((x, action_params), dim=1)
        
        ############### OLD VERSION BELOW ################
        
        #temp = torch.cat((state, action_params), dim=1)

        x1 = temp
        for i in range(len(self.layers)):
            x1 = F.relu(self.layers[i](x1))
            
        adv1 = self.adv_layers_1(x1)
        val1 = self.val_layers_1(x1)

        q_duel1 = val1 + adv1 - adv1.mean(dim=1, keepdim=True)

        x2 = temp
        for i in range(len(self.layers)):
            x2 = F.relu(self.layers[i](x2))
            
        adv2 = self.adv_layers_1(x2)
        val2 = self.val_layers_1(x2)
        q_duel2 = val2 + adv2 - adv2.mean(dim=1, keepdim=True)

        return q_duel1, q_duel2


class GaussianPolicy(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_layers=(256, 128, 64), action_space=None,
                 ):
        """

        :param state_dim:
        :param action_dim:
        :param hidden_layers:
        """
        super().__init__()
        
        # Convolutional layers to process state_dim with pooling layers
        self.conv1 = nn.Conv2d(in_channels=26, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the size of the output from the convolutional layers
        conv_output_size = 64 * (20 // 8) * (20 // 8) 
        state_dim = conv_output_size
        
        ############### OLD VERSION BELOW ################

        # initialize layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_layers[0]))
        
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            
        self.mean_layers = nn.Linear(hidden_layers[-1], action_dim)
        self.log_std_layers = nn.Linear(hidden_layers[-1], action_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.apply(init_)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        x = state
        
        ############### NEW VERSION IN BETWEEN ################
        
        # Pass the state through the convolutional and pooling layers
        state = torch.permute(state, (0,3,1,2))
        x = F.relu(self.bn1(self.conv1(state)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten the output from the convolutional layers
        #print("x shape debug: ",x.shape)
        x = x.reshape(x.size(0), -1)
        #x = x.view(x.size(0), -1)
        
        ############### NEW VERSION IN BETWEEN ################

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))
            
        # x = torch.nan_to_num(x, nan=0.0)
            
        mean = self.mean_layers(x)
        log_std = self.log_std_layers(x).clamp(-20, 2)
        return mean, log_std

#     def sample(self, state):
#         mean, log_std = self.forward(state)
        
#         std = log_std.exp()
#         # logger.info(f'std: {std}')
#         # logger.info(f'mean:'{mean}')
#         # normal = Normal(mean, std)
#         # x_t = normal.rsample()
#         # y_t = torch.tanh(x_t)
#         # action = y_t * self.action_scale + self.action_bias
#         # print('std:', std)
#         # print('mean:', mean)
#         noise = torch.randn_like(mean, requires_grad=True)
#         action = (mean + std * noise).tanh()

#         # log_prob = normal.log_prob(x_t)
#         # log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
#         # log_prob = log_prob.sum(1, keepdim=True)
#         # mean = torch.tanh(mean) * self.action_scale + self.action_bias
#         log_prob = log_std + np.log(np.sqrt(2 * np.pi)) + noise.pow(2).__mul__(0.5)
#         log_prob += (-action.pow(2) + 1.00000001).log()
#         return action, log_prob.sum(1, keepdim=True), mean

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        #logger.info(f'x_t: {x_t}')
        y_t = torch.tanh(x_t)
        #logger.info(f'y_t: {y_t}')
        #logger.info(f'action scale: {self.action_scale}')
        #logger.info(f'bias scale: {self.action_bias}')
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean