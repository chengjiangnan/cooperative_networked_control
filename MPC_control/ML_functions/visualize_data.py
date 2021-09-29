import torch
import torch.nn as nn

import numpy as np
import os, sys
import pandas
import argparse

ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR'] 
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')
sys.path.append(ROBOTICS_CODESIGN_DIR + '/MPC_control/')

SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/'

from utils import *

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    fig_ext = args.fig_ext

    BASE_DIR = SCRATCH_DIR + model_name

    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/results.pkl')
    PLOT_DIR = DATA_DIR

    T = data_dict['T']
    s_dim = data_dict['s_dim']

    train_dataset = data_dict['train_dataset']
    train_dataset = train_dataset.reshape(train_dataset.shape[0], T, s_dim).numpy()

    val_dataset = data_dict['val_dataset']
    val_dataset = val_dataset.reshape(val_dataset.shape[0], T, s_dim).numpy()

    name_to_dataset = {
                        "train": train_dataset,
                        "val": val_dataset
                      }

    full_window = list(range(T))
    for dataset_name in name_to_dataset.keys():
        dataset = name_to_dataset[dataset_name]

        time_series_q5 = np.percentile(dataset, 5, axis=0)
        time_series_q50 = np.percentile(dataset, 50, axis=0)
        time_series_q95 = np.percentile(dataset, 95, axis=0)

        for dim in range(s_dim):
            title_str = 'dim in s_dim: {}/{}'.format(dim, s_dim)
            plot_file = PLOT_DIR + '{}_s_dim_{}.{}'.format(dataset_name, dim, fig_ext)

            plt.plot(full_window, time_series_q5[:, dim], 'red', label='Q5')
            plt.plot(full_window, time_series_q50[:, dim], 'black', label='Q50')
            plt.plot(full_window, time_series_q95[:, dim], 'blue', label='Q95')

            plt.xlabel("time")
            plt.legend()
            plt.title(title_str)
            plt.savefig(plot_file)
            plt.close()
