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

    num_dimensions = 4
    x_dim = num_dimensions
    u_dim = num_dimensions
    s_dim = num_dimensions

    A = np.identity(num_dimensions)
    B = np.identity(num_dimensions)
    C = -np.identity(num_dimensions)
    Q = np.identity(num_dimensions)
    R = np.identity(num_dimensions)

    # for i in range(num_dimensions):
    #     C[i, i] = i+1

    W = 15
    H = 15
    
    T = 100

    u_ub = 0.95
    u_lb = -0.95

    iot_folder = ROBOTICS_CODESIGN_DIR + 'data/IoT_TPU/'

    data_types = ['train', 'val']

    # scenario_list = ['tamper_sensor', 'heat_shock', 'light_switch']
    scenario_list = [ 'heat_shock' ]

    column_list = ['pressure', 'humidity', 'temperature', 'ambient_light']

    raw_data_dict = OrderedDict()
    cnt_dict = OrderedDict()

    for data_type in data_types:
        base_folder = iot_folder + data_type + '/'
        aggre_data_df = pd.DataFrame()
        cnt = 0
        for scenario in scenario_list:
            csv_dir = base_folder + scenario
            cnt += len(list(enumerate(os.listdir(csv_dir))))
            for csv_num, csv_file in enumerate(os.listdir(csv_dir)):
                # print('[{}] {}'.format(csv_num, csv_file))
     
                data_df = pd.read_csv(csv_dir + '/' + csv_file)

                aggre_data_df = aggre_data_df.append(data_df[column_list])

                data_df = data_df[column_list]

                

        print(aggre_data_df.to_numpy().shape)
        raw_data_dict[data_type] = aggre_data_df
        cnt_dict[data_type] = cnt

    train_data = raw_data_dict['train']
    val_data = raw_data_dict['val']

    Q0 = train_data[column_list].quantile(0)
    Q1 = train_data[column_list].quantile(0.05)
    Q2 = train_data[column_list].quantile(0.95)
    Q3 = train_data[column_list].quantile(1)
    
    print("Q0: {}\nQ1: {}\nQ2: {}\nQ3: {}".format(Q0, Q1, Q2, Q3))


    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    min_max_scaler.fit(train_data)

    for i in range(len(column_list)):
        min_max_scaler.data_min_[i] = Q0[column_list[i]]
        min_max_scaler.data_max_[i] = Q3[column_list[i]]


    print("Min: {}".format(min_max_scaler.data_min_))
    print("Max: {}".format(min_max_scaler.data_max_))

    train_data = min_max_scaler.transform(train_data)
    val_data = min_max_scaler.transform(val_data)

    train_data = train_data.reshape(cnt_dict['train'], T, s_dim)
    val_data = val_data.reshape(cnt_dict['val'], T, s_dim)

    print(train_data.shape)
    print(val_data.shape)


    train_inputs = []
    train_outputs = []

    for i in range(cnt_dict['train']):
        for t in range(W-1, T-H+1):
            train_inputs.append(train_data[i, t-W+1:t+1, :].reshape(W*s_dim, 1))
            train_outputs.append(train_data[i, t:t+H, :].reshape(H*s_dim, 1))

    train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32)
    train_outputs_tensor = torch.tensor(train_outputs, dtype=torch.float32)

    print("train_inputs_tensor size: {}".format(train_inputs_tensor.size()))
    print("train_outputs_tensor size: {}".format(train_outputs_tensor.size()))

    _, train_dataloader = create_dataloader_from_tensors(
        train_inputs_tensor, train_outputs_tensor)

    train_dataset = torch.tensor(train_data.reshape(cnt_dict['train'], T*s_dim, 1), dtype=torch.float32)


    val_inputs = []
    val_outputs = []

    for i in range(cnt_dict['val']):
        for t in range(W-1, T-H+1):
            val_inputs.append(val_data[i, t-W+1:t+1, :].reshape(W*s_dim, 1))
            val_outputs.append(val_data[i, t:t+H, :].reshape(H*s_dim, 1))

    val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32)
    val_outputs_tensor = torch.tensor(val_outputs, dtype=torch.float32)

    print("val_inputs_tensor size: {}".format(val_inputs_tensor.size()))
    print("val_outputs_tensor size: {}".format(val_outputs_tensor.size()))

    _, val_dataloader = create_dataloader_from_tensors(
        val_inputs_tensor, val_outputs_tensor)
    val_dataset = torch.tensor(val_data.reshape(cnt_dict['val'], T*s_dim, 1), dtype=torch.float32)

    print("train_dataset size: {}".format(train_dataset.size()))
    print("val_dataset size: {}".format(val_dataset.size()))

    
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

    data_dict['u_ub'] = u_ub
    data_dict['u_lb'] = u_lb

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
