import torch
from torchvision import transforms
import pandas as pd
import numpy as np  
from skimage.feature import hog
from skimage.color import rgb2gray
from PIL import Image
import random
import functools
import cv2
from copy import deepcopy

from models import pt_feat_extractor

# load image and fix color (RGBA to RGB)
loader = lambda img_name : Image.open(img_name).convert('RGB')

# return indices of data used to create temporal stack of features
def past_indices(idx, history=2, dilation=8):
    '''
    history : number of frames to consider
    dilation : number of frames to skip + 1
    '''
    return list(np.maximum([idx - i*dilation for i in range(0,history)],0))

def resnet_features(image):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])        
        ])
    
    resnet_model = pt_feat_extractor

    # normalize it and convert to tensor
    if transform:
        image = transform(image).unsqueeze(0)

    # send image to correct device by checking where parameters of resnet model are stored
    image = image.to(next(resnet_model.parameters()).device)

    # extract resnet features
    resnet_features = resnet_model(image).squeeze(0)

    return resnet_features

def get_depth_features_VAE(data, idx):
    img_name = data['depth_addr'].iloc[idx]

    transform = transforms.Compose([transforms.Resize((64,64)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), \
            transforms.Normalize(mean=[0.5],std=[0.5])])

    image = Image.open(img_name)
    image = transform(image).unsqueeze(0)
    image = image.to(next(depth_model.parameters()).device)

    features = depth_model.encode(image)[0].view(-1)
    return features.cpu().detach().numpy()

def mode_checker(func):
    def wrapper(*args, **kwargs):
        # HDF5 building mode
        if len(kwargs) == 2:
            img_name = kwargs['data']['depth_addr'].iloc[kwargs['idx']]
            image = cv2.imread(img_name)
        # rollout mode
        else:
            image = args[0]
        return func(image)

    return wrapper

@mode_checker
def get_depth_features(image):
    image = cv2.resize(image, (64,64))
    image = rgb2gray(image)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8), 
        cells_per_block=(1, 1), feature_vector=True, visualize=True, multichannel=False)

    # print(fd.shape)

    return fd

def stack_frames(data, idx):        
    # stack frames
    features = torch.cat([
        resnet_features(
            loader(data['rgb_addr'].iloc[index])
            ) 
        for index in past_indices(idx)
        ], dim=0)

    # dummy data for testing purposes
    # features = torch.randn(256,28,28)

    return features.cpu().detach().numpy()

def get_gaze_data(data, idx):
    # use if averaging gaze data
    # a = data[['gaze_x','gaze_y']].iloc[past_indices(idx)].to_numpy()

    # use if only training for instantaneous gaze prediction
    a = data[['gaze_x','gaze_y']].iloc[idx].to_numpy()
    return a

def get_actions(data, idx):
    actions = data.loc[:,'act_roll':'act_yaw'].iloc[idx].to_numpy()

    return actions

def get_stacked_states(data, idx):    
    '''
    Notes:
    1) for stacking replace idx with past_indices(idx) and reshape
    2) to ignore GPS position start from pos_z instead of pos_x
    '''
    states = data.loc[:,'pos_z':'yaw_acc'].iloc[idx].to_numpy()

    return states

feature_generators = {
    'visual_features':stack_frames,
    'depth_features':get_depth_features,
    'kinematic_features':get_stacked_states,
    'gaze_coords':get_gaze_data,
    'actions':get_actions
}
