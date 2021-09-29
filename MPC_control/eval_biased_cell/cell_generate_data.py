import numpy as np
import pandas as pd
import os, sys
from collections import OrderedDict
import torch
from torch.utils.data import TensorDataset, DataLoader
import argparse

import math

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

    num_of_dimensions = 4
    x_dim = num_of_dimensions
    u_dim = num_of_dimensions
    s_dim = num_of_dimensions

    A = np.identity(num_of_dimensions)
    B = np.identity(num_of_dimensions)
    C = -np.identity(num_of_dimensions)
    Q_h = np.identity(num_of_dimensions)
    Q_b = 100 * np.identity(num_of_dimensions)
    R = np.identity(num_of_dimensions)

    W = 15
    H = 15

    T = 60
    num_samples = 35

    samples_per_day = 7
    num_of_days = int(num_samples / samples_per_day)
    T_full = T * samples_per_day

    data_list = np.zeros((num_of_days, T_full, s_dim))

    full_data_csv = ROBOTICS_CODESIGN_DIR + 'data/AAAI_cell/telstra/master_celx_rec_concat/MAST.CELX.{}.csv'
    cell_ids = [ '135718657', '136046083', '136046093', '136046103' ]

    for idx in range(len(cell_ids)):
        cell_id = cell_ids[idx]
        data_df = pd.read_csv(full_data_csv.format(cell_id))
        curr_list = data_df['CELLT_AGG_COLL_PER_TTI_DL'].tolist()
        for day_idx in range(num_of_days): 
            data_list[ day_idx, :, idx ] = curr_list[day_idx*(T_full+1):(day_idx+1)*(T_full+1)-1]

    data_list = data_list.reshape(num_samples, T, s_dim)
    data_list = data_list[1:, :, :]

    num_samples -= 1
    data_list = data_list.reshape(num_samples * T, s_dim)

    # min_max_scaler = MinMaxScaler()
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data_list = min_max_scaler.fit_transform(data_list)

    print("Min: {}".format(min_max_scaler.data_min_))
    print("Max: {}".format(min_max_scaler.data_max_))

    data_list = data_list.reshape(num_samples, T, s_dim)

    print(data_list)

    train_inputs = []
    train_outputs = []

    # for i in range(int(num_samples / 2)):
    for i in range(num_samples):
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

    # for i in range(int(num_samples / 2), num_samples):
    for i in range(num_samples):
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
    data_dict['Q_h'] = Q_h
    data_dict['Q_b'] = Q_b
    data_dict['R'] = R

    data_dict['W'] = W
    data_dict['H'] = H

    # length of the time series for evaluation
    data_dict['T'] = T

    write_pkl(fname = RESULTS_DIR + '/results.pkl', input_dict = data_dict)
