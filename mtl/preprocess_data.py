import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
import pandas as pd
import numpy as np  
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from skimage.feature import hog
from skimage.color import rgb2gray
from PIL import Image
import random
import os
from glob import glob
import sys
import logging
import functools
import h5py
from time import time
from ast import literal_eval
import matplotlib.pyplot as plt 
import cv2
from zlib import adler32
from copy import deepcopy

from models import *
from common_utils import *
from feat_gen import *

class DepthDataset(Dataset):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.files = [filename for filename in find_files(self.path, '.png') if 'depth' in filename]
        self.transform = transforms.Compose([transforms.Resize((64,64)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), \
            transforms.Normalize(mean=[0.5],std=[0.5])])

    def __len__(self):
        return len(self.files)

    @functools.lru_cache(maxsize=2000)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.files[idx]
        image = Image.open(img_name)
        image = self.transform(image)
        return image

class AirsimDataset(Dataset):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.hf = h5py.File('data.h5', 'r')

        self.input_feature_types = ['visual_features', 'depth_features', 'kinematic_features']
        self.feature_types = self.input_feature_types + ['gaze_coords', 'actions']

    def __len__(self):
        return len(self.data)

    @functools.lru_cache(maxsize=5000)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        env, command, episode, step_num = self.data[['env', 'command', 'epi_id', 'step_num']].iloc[idx]

        item = dict()

        for feature_type in self.feature_types:
            item.update({feature_type : torch.Tensor(self.hf[f'/{env}/{command}/{str(episode)}/{feature_type}'][step_num])})

        item.update({'command' : self.hf[f'/{env}'].attrs[command]})
        item.update({'camera_frame_file' : self.data['rgb_addr'].iloc[idx]})

        return item

def transform_df(df, transform):
    # transform is a list of tuples of (column_name, transform_fn) format
    mapper = DataFrameMapper(transform, df_out=True)

    df_ = mapper.fit_transform(df.copy())
    df[df_.columns] = df_

    return df

def create_df(folder=None, commands=[], feature_generators=feature_generators):

    # find csv files in task folders
    log_files = sorted(list(filter(lambda x: os.path.basename(x) == 'log.csv', find_files(folder, '.csv'))))

    # create dataframes from log files of all tasks
    dataframes = [pd.read_csv(file) for file in log_files]

    # initialise empty super dataframe
    df_merged = pd.DataFrame()

    for i, df in enumerate(dataframes):
        folder_basename = log_files[i].split(folder,1)[1]

        # checksum based deterministic hashing
        hash_fn = lambda x: adler32((folder_basename + str(x)).encode('utf-8'))
        df['epi_id'] = df['epi_num'].map(hash_fn)

        # add column with source folder info
        df['source'] = df['epi_num'].map(lambda x: f'{folder_basename},  Episode {str(x)}')

        # fix file addresses
        df['rgb_addr'] = df['rgb_addr'].map(lambda x: folder + '/' + x.split('data/',1)[1])
        df['depth_addr'] = df['depth_addr'].map(lambda x: folder + '/' + x.split('data/',1)[1])


        # strip whitespaces from columns
        df.rename(columns=lambda x: x.strip(), inplace=True)

        # merge dataframes
        df_merged = pd.concat([df_merged, df], ignore_index=True)
    
    df = df_merged

    # add environment attribute
    env_map = {
        'Fly to the top of utility pole':'urban',
        'Fly to red vehicle':'urban',
        'Fly to the nearest vehicle':'mountains',
        'Fly to the nearest truck':'mountains'
    }

    df = df.assign(env=[env_map[df['command'].iloc[i].strip()] for i in range(len(df))])
    
    # for debugging
    # pd.set_option('display.max_rows', None)
    # print(df.head)
    
    # 'a' flag appends to existing file and creates new if file doesn't exist.
    with h5py.File('data.h5', 'a') as hf:

        gb = df.groupby(['command'])
        for command in df.command.unique():

            df_command = gb.get_group(command)
            grouped_episodes = df_command.groupby(['epi_id'])

            for episode in df_command.epi_id.unique():
                data = grouped_episodes.get_group(episode)
                env, command, epi_id = data[['env','command','epi_id']].iloc[0]

                l1 = hf.require_group(env)
                l2 = l1.require_group(command)
                l3 = l2.require_group(str(epi_id))

                # add source folder metadata
                l3.attrs.update({'Source':data['source'].iloc[0]})
                # print(l3.attrs['Source'])

                # check if various feature types for this episode are already stored
                for feature_type in feature_generators.keys():
                    try: 
                        dataset_addr = f'{env}/{command}/{epi_id}/{feature_type}'
                        dataset = hf[dataset_addr]
                        print(f'already have {feature_type} data for episode id {epi_id} for task {command}')
                    except:
                        print(f'saving {feature_type} for episode id {epi_id} for task {command}')
                        features = np.array([feature_generators[feature_type](data=data, idx=idx) for idx in range(len(data))])
                        d = l3.create_dataset(f'{feature_type}', data=features)

    # categorically encode commands and include in environment metadata
    with h5py.File('data.h5', 'a') as hf:
        grouped_envs = df.groupby(['env'])

        for env in df.env.unique():
            env_data = grouped_envs.get_group(env)
            existing_mapping = hf[env].attrs

            # one-hot-encode the commands in each environment
            le = LabelEncoder()
            env_data = transform_df(env_data, [('command', le)])

            # store the command to category variable mapping
            new_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

            if set(new_mapping.keys()) == set(existing_mapping.keys()):
                assert new_mapping == existing_mapping, 'clean up data/ folder and start again with only what you need'

            existing_mapping.update(new_mapping)


    # save data file
    df.to_csv(f'{folder}/data.csv', index=False)

    # only include those tasks in dataframe as specified by user / wish to train on
    if len(commands) > 0:
        commands = [command.replace('_',' ') for command in commands]
        gb = df.groupby(['command'])

        # if you want to throw errors for unavailable tasks, comment line below
        # commands = list(filter(lambda x: x in df.command.unique(), commands))
        
        df = pd.concat([gb.get_group(key) for key in commands]).reset_index(drop=True)

    return df

def select_episodes(df, training_fractions = [0.8, 0.5, 0.4, 0.2, 0.1]):

    episodes = list(set(df['epi_id'].values))

    train_epi_ids_dict = dict()

    prev_sampled_episodes = deepcopy(episodes)

    for i, fraction in enumerate(training_fractions):

        num_episodes_to_sample = int(fraction*len(episodes))
        
        sampled_episodes = list(np.random.choice(prev_sampled_episodes, num_episodes_to_sample, replace=False))
        train_epi_ids_dict.update({fraction : sampled_episodes})
        
        prev_sampled_episodes = sampled_episodes

    # differing training test sizes (successively subsampled) but fixed test set
    biggest_train_set = train_epi_ids_dict[max(training_fractions)]

    test_epi_ids = list(set(episodes)^set(biggest_train_set))

    return train_epi_ids_dict, test_epi_ids

def get_dataframes(df, train_epi_ids, test_epi_ids):

    # create train/test dataframes by combining episodes from a shuffled list
    episode_groups = df.groupby(['epi_id'])
    df_train = pd.concat([episode_groups.get_group(i) for i in train_epi_ids])
    df_test = pd.concat([episode_groups.get_group(i) for i in test_epi_ids])

    return df_train, df_test
    

if __name__=='__main__':
    # display all rows
    # pd.set_option('display.max_rows', None)
    random.seed(0)
