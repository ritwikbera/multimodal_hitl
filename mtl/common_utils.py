import torch
import torch.nn as nn
import os 
from glob import glob
import numpy as np
import h5py
from itertools import product
import pickle
import signal
from contextlib import contextmanager
import time
import logging 

def get_device(model):
    return next(model.parameters()).device
    
def find_files(dirName, extension):
    files = list()

    for (dirpath, dirnames, filenames) in os.walk(dirName):
        files += [os.path.join(dirpath, file) for file in filenames if file.endswith(extension)]

    return files

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_logger(name):
    log_format = '%(asctime)s  %(name)8s  %(levelname)5s  %(message)s'
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                        filename='dev.log',
                        filemode='a+')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger(name)
    logger.addHandler(console)
    return logger 

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def load_best_model(model, path):
    # load best trained BC model, as per validation loss
    model_files = glob(path+'*.pth')

    # extract validation loss numbers from the model names
    val_losses = [float(filename[filename.find('val_loss=-')+len('val_loss=-'):filename.rfind('.pth')]) for filename in model_files]

    # select the one with the lowest validation loss
    model_file = model_files[np.argmin(val_losses)]

    model.load_state_dict(torch.load(model_file))
    print(f'Loaded {model_file} model.')


def safe_load(model, model_filename):
    pretrained_dict = torch.load(model_filename)
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict)

def get_instance(module, name, config, *args):
    try:
        return getattr(module, config[name]['type'])(*args, **config[name]['args'])
    except KeyError:
        return getattr(module, config[name]['type'])()

def delete_feature_type(file='data.h5', feature_type='depth_features'):
    def visitor_func(name, node):
        if isinstance(node, h5py.Dataset):
            if feature_type not in name:
                print(f'Transferring {name}')
                new_file.create_dataset(name, data=f[name][:])

    f = h5py.File(file, 'a')
    new_file = h5py.File('new.h5','a')
    f.visititems(visitor_func)

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        yield dict(zip(keys, instance))

def save_outputs(model, device, train_loader, val_loader):

    outputs, labels = torch.Tensor().to(device), torch.Tensor().to(device)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        outputs = torch.cat((outputs, model(data.to(device))),dim=0)
        labels = torch.cat((labels, target.to(device)), dim=0)

    split = labels.size(0)

    for batch_idx, (data, target) in enumerate(val_loader):
        outputs = torch.cat((outputs, model(data.to(device))),dim=0)
        labels = torch.cat((labels, target.to(device)), dim=0)

    outputs = outputs.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    np.savez('outputs.npz', outputs=outputs, labels=labels, split=split)

if __name__=='__main__':
    delete_feature_type()