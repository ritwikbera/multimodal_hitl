import numpy as np 
import torch
from torch.utils.data import Sampler, Dataset, WeightedRandomSampler
from torch.utils.data.sampler import BatchSampler
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

import sys


def get_BC_sampler(df_train):

    # visualize training data
    num_bins = 25
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    df_train.hist(bins=num_bins, column=['act_roll', 'act_pitch', 'act_throttle', 'act_yaw'], ax=ax1)
    fig.savefig(f'actions.png')

    # binning of action data for sampling purposes
    vals = df_train['act_yaw']
    statistic, _, binnumber= binned_statistic(vals, vals, statistic='count', bins=num_bins)
    
    # WeightedRandomSampler does not need weights to sum upto one
    bin_weights = 1.0/statistic
    df_train['weight'] = bin_weights[binnumber-1]

    samples_weight = torch.from_numpy(df_train['weight'].to_numpy())
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return sampler

class Sampler_(Sampler):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class StratifiedSampler(Sampler_):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)