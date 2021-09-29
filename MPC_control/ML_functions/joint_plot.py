import numpy as np
import os, sys
import pandas
import argparse

ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR'] 
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')
sys.path.append(ROBOTICS_CODESIGN_DIR + '/MPC_control/')

SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/'

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict
from utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_types', type=str)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()
    model_name = args.model_name
    train_types = args.train_types.split(',')
    fig_ext = args.fig_ext

    BASE_DIR = SCRATCH_DIR + model_name
    
    cost_var = r'Control Cost $J^\mathrm{c}$'
    recon_var = r'Average Forecasting Error (MSE)'
    train_var = r'Train Loss'
    size_var = r"Bottleneck Dimension $Z$"
    train_type_var = 'Policy'

    PLOT_DIR = BASE_DIR + '/test_results/'

    cost_results_df = pandas.DataFrame()
    train_results_df = pandas.DataFrame()
    recon_results_df = pandas.DataFrame()

    plot_optimal_line = False
    if "optimal" in train_types:
        train_types.remove("optimal")
        plot_optimal_line = True

        result_dict = load_pkl(PLOT_DIR + '/optimal/optimal.pkl')
        optimal_cost = result_dict['cost_mean']

        print("optimal_cost is: {}".format(optimal_cost))

    for train_type in train_types:
        
        plot_train_type = train_type_variable_to_name[train_type]

        TEST_DATA_DIR = PLOT_DIR + train_type

        pkl_file_list = [x for x in os.listdir(TEST_DATA_DIR) if '.pkl' in x]

        for pkl_file in pkl_file_list:
            # print('pkl file: ', pkl_file)       
            result_dict = load_pkl(TEST_DATA_DIR + '/' + pkl_file)
            # print(' ')
            s_dim = result_dict['s_dim']
            z_dim = result_dict['z_dim']
            cost_array = result_dict['cost_array']
            train_loss = result_dict['train_loss']
            recon_loss = result_dict['recon_loss']
            W = result_dict['W']
            H = result_dict['H']

            # attach to cost_results_df
            basic_results_df = pandas.DataFrame()
            basic_results_df[cost_var] = cost_array
            basic_results_df[size_var] = [ z_dim ] * len(cost_array)
            basic_results_df[train_type_var] = [ plot_train_type ] * len(cost_array)

            cost_results_df = cost_results_df.append(basic_results_df)

            # attach to train_results_df
            # if train_type == 'task_aware_first_control':
            #     continue
            basic_results_df = pandas.DataFrame()
            basic_results_df[train_var] = [ train_loss ]
            basic_results_df[size_var] = [ z_dim ]
            basic_results_df[train_type_var] = [ plot_train_type ]

            train_results_df = train_results_df.append(basic_results_df)

            # attach to recon_results_df
            basic_results_df = pandas.DataFrame()
            basic_results_df[recon_var] = [ recon_loss ]
            basic_results_df[size_var] = [ z_dim ]
            basic_results_df[train_type_var] = [ plot_train_type ]

            recon_results_df = recon_results_df.append(basic_results_df)


    # title_str = 'W = {}, H = {}, s_dim = {}'.format(W, H, s_dim)

    plot_file = PLOT_DIR + '/{}_cost_bottleneck.{}'.format(model_name, fig_ext)

    sns.pointplot(x=size_var, y=cost_var, data=cost_results_df, hue=train_type_var, palette=draw_color_palette)
    if plot_optimal_line:
        plt.axhline(y = optimal_cost, linewidth = 2.0, ls = '--', color = 'black', label = 'Optimal')
    plt.legend()
    # plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()

    plot_file = PLOT_DIR + '/{}_recon_loss.{}'.format(model_name, fig_ext)
    sns.pointplot(x=size_var, y=recon_var, data=recon_results_df, hue=train_type_var, palette=draw_color_palette)
    plt.legend()
    # plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()

    plot_file = PLOT_DIR + '/{}_train_loss.{}'.format(model_name, fig_ext)
    sns.pointplot(x=size_var, y=train_var, data=train_results_df, hue=train_type_var, palette=draw_color_palette)
    plt.legend()
    # plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()
