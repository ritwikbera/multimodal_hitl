import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import mplcursors
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import ConcatDataset, TensorDataset, Dataset, DataLoader, WeightedRandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, Rprop
import torchvision.models as models
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.manifold import TSNE
from sklearn_pandas import DataFrameMapper
from scipy.stats import binned_statistic
from skimage.feature import hog
from skimage.color import rgb2gray
import seaborn as sns
from PIL import Image
import random
import os
import sys
from glob import glob
import math
import json
import gc
from datetime import datetime
import logging
import argparse
import h5py
import pickle
from copy import deepcopy
from inspect import getfullargspec
from ast import literal_eval
from zlib import adler32
from itertools import product
import mlflow
from time import time
from collections import defaultdict
import signal
from contextlib import contextmanager

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping, ModelCheckpoint