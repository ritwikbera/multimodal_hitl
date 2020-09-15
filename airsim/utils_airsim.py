"""utils_airsim.py
Utility functions for data collection in AirSim.
"""
import os
import pygame
import numpy as np
import pandas as pd
import math
import airsim
import time
import matplotlib.pyplot as plt
import threading
from subprocess import Popen, PIPE
import sys
from glob import glob 
from ast import literal_eval
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as pt_models
from torchvision import transforms
import logging

models_path = os.getcwd().split('airsim')[0]+'mtl/'
sys.path.insert(0, models_path)
import models as mtl_models
from preprocess_data import *
from common_utils import *
from feat_gen import *

from PIL import Image

class JoystickAgent(object):
    """ Reads an Xbox joystick to control AirSim vehicles.
    """
    def __init__(self):
        self.name = 'JoystickAgent'

        # setup xbox joystick using pygame
        pygame.init()

        # Initialize the connected joysticks
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        print('Joysticks connected: {}'.format(joystick_count))

        # only use first connected
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print('Using joystick: {}'.format(self.joystick.get_name()))
       

    def act(self, ob=None, reward=None, done=None):
        """ Reads and return left and right sticks."""
        pygame.event.pump()

        # read left stick (vertical)
        left_stick_vert = self.joystick.get_axis(1)
        left_stick_horz = self.joystick.get_axis(0)

        # read right stick (horizontal)
        right_stick_horz = self.joystick.get_axis(3)
        right_stick_vert = -self.joystick.get_axis(4)

        # concatenate joystick values
        action = np.array([
            right_stick_horz, right_stick_vert,
            left_stick_vert, left_stick_horz])

        # apply deadband
        action = self._apply_deadband(action)

        # check for reset button
        reset_button = self.joystick.get_button(0)
        if reset_button == 1:
            self.reset_pressed = True
        else:
            self.reset_pressed = False

        return action

    def _check_intervention(self):
        """Checks if intervention button (top left binary button) is being
        pressed and returns True or False."""
        pygame.event.pump()
        return self.joystick.get_button(4) 

    def _apply_deadband(self, action):
        """Create a deadband to smooth the data coming from the joysticks."""
        dbv = 0.05 # deadband value
        
        # roll
        if action[0] < dbv and action[0] > -dbv:
            action[0] = 0.0

        # pitch
        if action[1] < dbv and action[1] > -dbv:
            action[1] = 0.0

        # throttle
        if action[2] < dbv and action[2] > -dbv:
            action[2] = 0.0

        # yaw
        if action[3] < dbv and action[3] > -dbv:
            action[3] = 0.0

        return action


    def close(self):
        """ Stop any thread (if any) or save additional data (if any)"""
        pygame.quit()
        print('Joystick connection closed.')


class AirSimBCAgent(object):
    """ AirSim behavior cloning agent
    """
    def __init__(self, exp_name, task=None, env='mountains'):
        self.name = 'AirSimBCAgent'
        self.task = task

        # check if using gaze or not, then create model
        print('Loading model from experiment:', exp_name)

        config_file = f'../mtl/{exp_name}/config.txt'
        config = literal_eval(open(config_file, 'r').read())
        
        self.model = getattr(mtl_models, config['model_type'])(**config['model_params'])
        
        load_best_model(self.model, f'../mtl/{exp_name}/')
        self.model.eval() # to use learnt mean and std (by batchnorm)
        self.device = get_device(self.model)

        self.feature_buffer = list()

        # # mean and std for kinematic features (from training data)
        # self.stats = literal_eval(open('../mtl/data/train_stats.txt', 'r').read())
        # self.mean = torch.Tensor(list(self.stats['mean'].values()))
        # self.std = torch.Tensor(list(self.stats['std'].values()))
        
        self.hdf_file = '../mtl/data.h5'
        self.hf = h5py.File(self.hdf_file, 'r')
        self.mapping = self.hf[f'/{env}'].attrs
        try:
            _ = self.mapping[self.task]
        except KeyError:
            print('Incorrect environment, change env input for the agent')

        # setup stacked frames
        self.history = 2
        self.dilation = 8
        self.buffer_size = 1 + (self.history-1)*self.dilation       

    def act(self, ob=None, reward=None, done=None):
        """ Pass observation through model and return action."""
        # agent never resets the task
        self.reset_pressed = False

        # postprocess observation:
        # extract resnet features
        image = Image.fromarray(np.copy(ob['img_rgb']))
        depth_map = np.copy(ob['img_depth'])

        visual_features = resnet_features(image).to(self.device).unsqueeze(0)
        depth_features = torch.Tensor(get_depth_features(depth_map)).to(self.device).view(1,-1)

        # first observation recorded
        if len(self.feature_buffer) == 0:
            self.feature_buffer = [visual_features]*self.buffer_size
        else:
            self.feature_buffer.pop(0)  
            self.feature_buffer.append(visual_features)

        # features are added in the buffer at every time step but used as per specified dilation only
        stacked_resnet_features = torch.cat(self.feature_buffer[::self.dilation], dim=1)


        # use only a few states from the full states captured in AirSim
        state = ob['states_bc'][0]
        pitch = ob['states_bc'][1]
        roll = ob['states_bc'][2]
        yaw = ob['states_bc'][3]
        states_bc = torch.Tensor([
            # state.kinematics_estimated.position.x_val,
            # state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val,
            roll,
            pitch,
            yaw,
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val,
            state.kinematics_estimated.angular_velocity.x_val,
            state.kinematics_estimated.angular_velocity.y_val,
            state.kinematics_estimated.angular_velocity.z_val,
            state.kinematics_estimated.linear_acceleration.x_val,
            state.kinematics_estimated.linear_acceleration.y_val,
            state.kinematics_estimated.linear_acceleration.z_val,
            state.kinematics_estimated.angular_acceleration.x_val,
            state.kinematics_estimated.angular_acceleration.y_val,
            state.kinematics_estimated.angular_acceleration.z_val]).view(1,-1)

        # states_bc = (states_bc - self.mean)/self.std
        # prepare input tensor

        command = torch.LongTensor([self.mapping[self.task]]).to(self.device)

        inputs = {'visual_features':stacked_resnet_features,'depth_features':depth_features,'kinematic_features':states_bc,'command':command}
        # return action from observation
        with torch.no_grad():
            pred = self.model(inputs) 

        # passing action together with gaze prediction and will handle splitting
        # between gaze and action inside the env. this way, we can easily log the predicted gazes
        action = pred

        # # action is always last in multi-output networks
        # pred = pred[-1] if type(pred) in [tuple, list] else pred
        # action = pred.cpu().detach().numpy()[0]

        return action

    def close(self):
        """ Stop any thread (if any) or save additional data (if any)"""
        print('Done with BC agent.')


def postprocess_airsim_dataset(dataset_addr):
    """Splits AirSim dataset in training and validation and compute
    normalization metrics for training.
    """
    # load dataset
    dataset = pd.read_csv(os.path.join(dataset_addr,'log.csv'))

    # compute normalization metrics for state variables
    states_actions = dataset.iloc[:, 9:-3]
    states_actions_mean = states_actions.mean()
    states_actions_std = states_actions.std()
    states_actions_norm = (states_actions-states_actions_mean)/states_actions_std

    return states_actions_norm, states_actions_mean, states_actions_std



if __name__ == '__main__':
    postprocess_airsim_dataset('data/copy_fly_tree/')

