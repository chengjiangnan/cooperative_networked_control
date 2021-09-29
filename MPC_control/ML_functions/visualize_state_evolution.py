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
    parser.add_argument('--train_types', type=str)
    parser.add_argument('--z_dims', type=str)
    parser.add_argument('--sample_idx', type=int)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    z_dims = args.z_dims.split(',')
    sample_idx = args.sample_idx
    fig_ext = args.fig_ext

    train_types = args.train_types.split(',')


    BASE_DIR = SCRATCH_DIR + model_name

    PLOT_DIR = BASE_DIR + '/visualize_state_evolution/'
    remove_and_create_dir(PLOT_DIR)

    for subfolder_name in [ 'state_evolution', 'control_evolution' ]:
        remove_and_create_dir('{}/{}/'.format(PLOT_DIR, subfolder_name))
        for z_dim_str in z_dims: 
            remove_and_create_dir('{}/{}/z_{}/'.format(PLOT_DIR, subfolder_name, z_dim_str))

    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/results.pkl')

    loss_fn = torch.nn.MSELoss()

    x_array_dict = OrderedDict()
    u_array_dict = OrderedDict()


    T = data_dict['T']
    x_dim = data_dict['x_dim']
    u_dim = data_dict['u_dim']
    s_dim = data_dict['s_dim']
    W = data_dict['W']
    H = data_dict['H']
    
    # real T used for state involvement
    num_intervals = T-H-W+2

    if "optimal" in train_types:
        train_types.remove("optimal")
        train_types = ["optimal"] + train_types

    for z_dim_str in z_dims:
        z_dim = int(z_dim_str)
        for train_type in train_types:

            RESULT_DIR = BASE_DIR + '/test_results/' + str(train_type)
            if train_type == "optimal":
                result_dict = load_pkl(RESULT_DIR + '/optimal.pkl')
            else:
                result_dict = load_pkl(RESULT_DIR + '/z_' + str(z_dim) + '.pkl')

            x_array_dict[train_type] = result_dict['x_array'][sample_idx, :, :]
            u_array_dict[train_type] = result_dict['u_array'][sample_idx, :, :]

        for dim in range(x_dim):
            print(" ")
            print("dim: {}".format(dim))
            # state evolution
            plot_file = '{}/state_evolution/z_{}/'.format(PLOT_DIR, z_dim_str)  + \
                        'state_evolution_sample_{}_state_{}'.format(sample_idx, dim) + '.{}'.format(fig_ext)

            for train_type in train_types:
                plt.plot(list(range(num_intervals+1)), x_array_dict[train_type][:, dim], 
                         color=train_type_variable_to_color[train_type],
                         label=train_type_variable_to_name[train_type], lw=2.0, ls='-')

                # print("train_type: {}; x_array: {}".format(train_type, x_array_dict[train_type][:, dim]))

            plt.xlabel("Time Index")
            plt.ylabel(r"{}".format("$x({})$".format(dim)))

            plt.legend()
            plt.savefig(plot_file)
            plt.close()

        for dim in range(u_dim):
            print(" ")
            print("dim: {}".format(dim))
            
            # control evolution
            plot_file = '{}/control_evolution/z_{}/'.format(PLOT_DIR, z_dim_str)  + \
                        'control_evolution_sample_{}_control_{}'.format(sample_idx, dim) + '.{}'.format(fig_ext)

            for train_type in train_types:
                plt.plot(list(range(num_intervals)), u_array_dict[train_type][:, dim], 
                         color=train_type_variable_to_color[train_type],
                         label=train_type_variable_to_name[train_type], lw=2.0, ls='-')

                # print("train_type: {}; u_array: {}".format(train_type, u_array_dict[train_type][:, dim]))

            plt.xlabel("Time Index")
            plt.ylabel(r"{}".format("$u({})$".format(dim)))

            plt.legend()
            plt.savefig(plot_file)
            plt.close()
