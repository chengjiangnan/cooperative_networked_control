import numpy as np
import pandas as pd
import os, sys
from collections import OrderedDict
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse

import math

from scipy import signal

ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR'] 
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')

from textfile_utils import *
from plotting_utils import *
from sklearn.preprocessing import MinMaxScaler


DEFAULT_PARAMS = {'batch_size':  1024, 'shuffle': True, 'num_workers': 1, 'pin_memory': True}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

np.random.seed(0)

def create_dataloader_from_tensors(inputs_tensor, outputs_tensor, params = DEFAULT_PARAMS):

    tensor_dataset = TensorDataset(inputs_tensor, outputs_tensor)
    tensor_dataloader = DataLoader(tensor_dataset, **params)
    
    return tensor_dataset, tensor_dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    args = parser.parse_args()
    
    model_name = args.model_name

    SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/' + model_name

    pjm_folder = ROBOTICS_CODESIGN_DIR + 'data/PJM/'

    # NI has only 2004-2010 data
    zones = [ 'AEP', 'COMED', 'DAYTON', 'DEOK', 'DOM', 'DUQ', 'EKPC', 'FE']

    num_dimensions = len(zones)

    x_dim = num_dimensions
    u_dim = num_dimensions
    s_dim = num_dimensions

    A = np.identity(num_dimensions)
    B = np.identity(num_dimensions)
    C = -np.identity(num_dimensions)
    Q = np.identity(num_dimensions)
    R = np.identity(num_dimensions)


    W = 24
    H = 24

    T = 168
    num_samples = 30

    zone_dfs = []

    for zone in zones:
        csv_path = pjm_folder + '{}_hourly.csv'.format(zone)
        basic_df = pd.read_csv(csv_path, parse_dates=['Datetime'])
        basic_df = basic_df[basic_df.Datetime.dt.year == 2017]
        basic_df = basic_df.set_index('Datetime')
        zone_dfs.append(basic_df)
        print(basic_df)
        # print(list(basic_df))
        

    data_df = pd.concat(zone_dfs, axis=1)

    print(data_df)


    data_list = np.zeros((num_samples*T, s_dim))

    for i in range(s_dim):
        column_name = "{}_MW".format(zones[i])
        data_list[ :, i] = data_df[column_name].to_list()[:num_samples*T]
    
    print(data_list)

    # min_max_scaler = MinMaxScaler()
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data_list = min_max_scaler.fit_transform(data_list)

    print("Min: {}".format(min_max_scaler.data_min_))
    print("Max: {}".format(min_max_scaler.data_max_))

    data_list = data_list.reshape(num_samples, T, s_dim)

    print(data_list)

    train_inputs = []
    train_outputs = []

    for i in range(int(num_samples / 2)):
        for t in range(W-1, T-H+1):
            train_inputs.append(data_list[i, t-W+1:t+1, :].reshape(W*s_dim, 1))
            train_outputs.append(data_list[i, t:t+H, :].reshape(H*s_dim, 1))

    train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32)
    train_outputs_tensor = torch.tensor(train_outputs, dtype=torch.float32)

    _, train_dataloader = create_dataloader_from_tensors(train_inputs_tensor, train_outputs_tensor)

    train_time_series = data_list[:int(num_samples/2), : ,:].reshape(int(num_samples/2), T*s_dim, 1)
    train_dataset = torch.tensor(train_time_series, dtype=torch.float32)


    val_inputs = []
    val_outputs = []

    for i in range(int(num_samples / 2), num_samples):
        for t in range(W-1, T-H+1):
            val_inputs.append(data_list[i, t-W+1:t+1, :].reshape(W*s_dim, 1))
            val_outputs.append(data_list[i, t:t+H, :].reshape(H*s_dim, 1))


    val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
    val_outputs_tensor = torch.tensor(val_outputs, dtype=torch.float32)

    _, val_dataloader = create_dataloader_from_tensors(val_inputs_tensor, val_outputs_tensor)

    val_time_series = data_list[int(num_samples/2):num_samples, : ,:].reshape(int(num_samples/2), T*s_dim, 1)
    val_dataset = torch.tensor(val_time_series, dtype=torch.float32)

    RESULTS_DIR = SCRATCH_DIR + '/dataset/'
    remove_and_create_dir(RESULTS_DIR)

    data_dict = OrderedDict()

    data_dict['train_inputs'] = train_inputs_tensor
    data_dict['train_outputs'] = train_outputs_tensor
    data_dict['train_dataloader'] = train_dataloader
    data_dict['train_dataset'] = train_dataset

    data_dict['val_inputs'] = val_inputs_tensor
    data_dict['val_outputs'] = val_outputs_tensor
    data_dict['val_dataloader'] = val_dataloader
    data_dict['val_dataset'] = val_dataset

    data_dict['x_dim'] = x_dim
    data_dict['u_dim'] = u_dim
    data_dict['s_dim'] = s_dim

    data_dict['A'] = A
    data_dict['B'] = B
    data_dict['C'] = C
    data_dict['Q'] = Q
    data_dict['R'] = R

    data_dict['W'] = W
    data_dict['H'] = H

    # length of the time series for evaluation
    data_dict['T'] = T

    write_pkl(fname = RESULTS_DIR + '/results.pkl', input_dict = data_dict)
