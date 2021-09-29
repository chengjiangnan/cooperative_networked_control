import torch
import torch.nn as nn

import numpy as np
import os, sys
import pandas
import argparse

ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR'] 
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')

SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/'

sys.path.append(ROBOTICS_CODESIGN_DIR + '/MPC_control/')

from utils import *

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--forecaster_name', type=str)
    parser.add_argument('--forecaster_hidden_dim', type=int)
    parser.add_argument('--train_types', type=str)
    parser.add_argument('--z_dims', type=str)
    parser.add_argument('--sample_idx', type=int)
    parser.add_argument('--time_idx', type=int)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    forecaster_name = args.forecaster_name
    forecaster_hidden_dim = args.forecaster_hidden_dim
    z_dims = args.z_dims.split(',')
    sample_idx = args.sample_idx # which sample in the val_dataset to use
    time_idx = args.time_idx # the starting time for forecasting
    train_types = args.train_types.split(',')
    fig_ext = args.fig_ext

    BASE_DIR = SCRATCH_DIR + model_name

    PLOT_DIR = BASE_DIR + '/visualize_forecasts/'
    remove_and_create_dir(PLOT_DIR)

    for z_dim_str in z_dims: 
        remove_and_create_dir('{}/z_{}/'.format(PLOT_DIR, z_dim_str))


    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/results.pkl')

    model_params = OrderedDict()
    model_params["W"] = data_dict['W']
    model_params["H"] = data_dict['H']
    model_params["s_dim"] = data_dict['s_dim']
    model_params["hidden_dim"] =  forecaster_hidden_dim

    T = data_dict['T']
    s_dim = data_dict['s_dim']
    W = data_dict['W']
    H = data_dict['H']
    val_dataset = data_dict['val_dataset'][sample_idx, :, :]
    val_dataset = val_dataset.to(device)
    val_dataset = val_dataset.reshape(T, s_dim)

    val_dataset_np = val_dataset.cpu().detach().numpy()

    # print(val_dataset.size())

    if "optimal" in train_types:
        train_types.remove("optimal")
    
    s_future_hat_np = {}

    x_label_name = "Time Index"
    shift = -(W-1)

    full_window = list(range(T))
    past_window = list(range(time_idx+1-W, time_idx+1))
    future_window = list(range(time_idx, time_idx+H))

    #  for plotting they need to be shifted
    shifted_full_window = [ shift+t for t in full_window ]
    shifted_past_window = [ shift+t for t in past_window ]
    shifted_future_window = [ shift+t for t in future_window ]

    # uniform_ylim = (-1.5, 1.0)
    # uniform_ylim = (-2.2, 1.8)
    # uniform_ylim = (-0.8, 1.2)

    for z_dim_str in z_dims:
        z_dim = int(z_dim_str)
        for train_type in train_types:

            MODEL_DIR = BASE_DIR + '/trained_models/' + str(train_type)

            model_path = MODEL_DIR + '/z_{}.pt'.format(z_dim)
            model_save_dict = torch.load(model_path)

            model_params["z_dim"] =  z_dim

            forecaster = init_forecaster(forecaster_name, model_params)
            forecaster.load_state_dict(model_save_dict["forecaster_state_dict"]) 
            forecaster.eval()

            s_past = val_dataset[past_window, :].reshape(1, W * s_dim)
            s_future_hat = forecaster(s_past).reshape(H, s_dim)
            s_future_hat_np[train_type] = s_future_hat.cpu().detach().numpy()

        for dim in range(s_dim):
            title_str = 'dim in s_dim: {}/{}'.format(dim, s_dim)
            plot_file = '{}/z_{}/'.format(PLOT_DIR, z_dim_str) + \
                        ('forecasts_task_aware_sample_{}_time_{}_signal_{}.{}'
                         .format(sample_idx, time_idx, dim, fig_ext))
            
            plt.plot(shifted_full_window, val_dataset_np[:, dim], 'red', label='True', alpha=0.2)
            plt.plot(shifted_past_window, val_dataset_np[past_window, dim], 'blue', label='Past Input', lw=2.0)
            plt.plot(shifted_future_window, val_dataset_np[future_window, dim], 'black', label='Future', lw=2.0)
            for train_type in train_types:
                if train_type != "task_aware_first_control":
                    continue
                plt.plot(shifted_future_window, s_future_hat_np[train_type][:, dim], 
                         color=train_type_variable_to_color[train_type],
                         label=train_type_variable_to_name[train_type], lw=2.0, ls='--')

            plt.xlabel(x_label_name)
            plt.ylabel(r"{}".format("$s({})$".format(dim)))
            # plt.ylim((0, 1))
            # plt.ylim((-1, 1))
            # plt.ylim(uniform_ylim)
            plt.legend()
            # plt.title(title_str)
            plt.savefig(plot_file)
            plt.close()

            # this piece of code plots figure without task-aware policy
            plot_file = '{}/z_{}/'.format(PLOT_DIR, z_dim_str) + \
                        ('forecasts_no_task_aware_sample_{}_time_{}_signal_{}.{}'
                         .format(sample_idx, time_idx, dim, fig_ext))
            
            plt.plot(shifted_full_window, val_dataset_np[:, dim], 'red', label='True', alpha=0.2)
            plt.plot(shifted_past_window, val_dataset_np[past_window, dim], 'blue', label='Past Input', lw=2.0)
            plt.plot(shifted_future_window, val_dataset_np[future_window, dim], 'black', label='Future', lw=2.0)
            for train_type in train_types:
                if train_type == "task_aware_first_control":
                    continue
                plt.plot(shifted_future_window, s_future_hat_np[train_type][:, dim], 
                         color=train_type_variable_to_color[train_type],
                         label=train_type_variable_to_name[train_type], lw=2.0, ls='--')

            plt.xlabel(x_label_name)
            plt.ylabel(r"{}".format("$s({})$".format(dim)))
            # plt.ylim((0, 1))
            # plt.ylim((-1, 1))
            # plt.ylim(uniform_ylim)
            plt.legend()
            # plt.title(title_str)
            plt.savefig(plot_file)
            plt.close()


   
    
