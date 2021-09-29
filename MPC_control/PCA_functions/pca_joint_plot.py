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


cnt = 0
draw_color_palette = {}

def append_to_dataframe(TEST_DATA_DIR, train_type, cost_results_df, fcst_results_df, ctrl_results_df):
    pkl_file_list = [x for x in os.listdir(TEST_DATA_DIR) if '.pkl' in x]

    for pkl_file in pkl_file_list:
        # print('pkl file: ', pkl_file)    
        result_dict = load_pkl(TEST_DATA_DIR + '/' + pkl_file)
        # print(' ')
        s_dim = result_dict['s_dim']
        z_dim = result_dict['z_dim']
        cost_array = result_dict['cost_array']
        fcst_diff = result_dict['fcst_diff']
        ctrl_diff = result_dict['ctrl_diff'] 
        H = result_dict['H']

        # attach to cost_results_df
        basic_results_df = pandas.DataFrame()
        basic_results_df[cost_var] = cost_array
        basic_results_df[size_var] = [ z_dim ] * len(cost_array)
        basic_results_df[train_type_var] = [ plot_train_type ] * len(cost_array)

        cost_results_df = cost_results_df.append(basic_results_df)

        # if train_type != "task_aware_first_control":
        basic_results_df = pandas.DataFrame()
        basic_results_df[fcst_var] = [ fcst_diff ]
        basic_results_df[size_var] = [ z_dim ]
        basic_results_df[train_type_var] = [ plot_train_type ]

        fcst_results_df = fcst_results_df.append(basic_results_df)

        basic_results_df = pandas.DataFrame()
        basic_results_df[ctrl_var] = [ ctrl_diff ]
        basic_results_df[size_var] = [ z_dim ]
        basic_results_df[train_type_var] = [ plot_train_type ]

        ctrl_results_df = ctrl_results_df.append(basic_results_df)

    return cost_results_df, fcst_results_df, ctrl_results_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_types', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()
    model_name = args.model_name
    train_types = args.train_types.split(',')
    weights = args.weights.split(',')
    weights = [float(weight) for weight in weights]
    fig_ext = args.fig_ext

    BASE_DIR = SCRATCH_DIR + model_name
    
    cost_var = r'Control Cost $J^\mathrm{c}$'
    fcst_var = r'Forecasting Error $J^\mathrm{F}$ (MSE)'
    ctrl_var = r'Control Error (MSE)'
    size_var = r"Bottleneck Dimension $Z$"
    train_type_var = 'Policy'

    PLOT_DIR = BASE_DIR + '/pca_results/'

    cost_results_df = pandas.DataFrame()
    fcst_results_df = pandas.DataFrame()
    ctrl_results_df = pandas.DataFrame()

    plot_optimal_line = False
    if "optimal" in train_types:
        train_types.remove("optimal")
        plot_optimal_line = True

        result_dict = load_pkl(PLOT_DIR + '/optimal/optimal.pkl')
        optimal_cost = result_dict['cost_mean']

        print("optimal_cost is: {}".format(optimal_cost))

    for train_type in train_types:
        
        
        if train_type == 'weighted':
            for weight in weights:
                TEST_DATA_DIR = PLOT_DIR + train_type + '_' + str(weight)
                plot_train_type = train_type_variable_to_name[train_type] + r', $\lambda^\mathrm{F}$=' + str(weight)
                draw_color_palette[plot_train_type] = palette_colors[cnt]
                cnt += 1
                cost_results_df, fcst_results_df, ctrl_results_df = append_to_dataframe(
                    TEST_DATA_DIR, train_type, cost_results_df, fcst_results_df, ctrl_results_df)
        else:
            TEST_DATA_DIR = PLOT_DIR + train_type
            plot_train_type = train_type_variable_to_name[train_type]
            draw_color_palette[plot_train_type] = palette_colors[cnt]
            cnt += 1
            cost_results_df, fcst_results_df, ctrl_results_df = append_to_dataframe(
                TEST_DATA_DIR, train_type, cost_results_df, fcst_results_df, ctrl_results_df)


    plot_file = PLOT_DIR + '/{}_cost_bottleneck.{}'.format(model_name, fig_ext)

    sns.pointplot(x=size_var, y=cost_var, data=cost_results_df, hue=train_type_var, palette=draw_color_palette)
    if plot_optimal_line:
        plt.axhline(y = optimal_cost, linewidth = 2.0, ls = '--', color = 'black', label = 'Optimal')
    plt.legend()
    # plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()

    plot_file = PLOT_DIR + '/{}_fcst_MSE.{}'.format(model_name, fig_ext)
    sns.pointplot(x=size_var, y=fcst_var, data=fcst_results_df, hue=train_type_var, palette=draw_color_palette)
    plt.legend()
    # plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()

    plot_file = PLOT_DIR + '/{}_ctrl_MSE.{}'.format(model_name, fig_ext)
    sns.pointplot(x=size_var, y=ctrl_var, data=ctrl_results_df, hue=train_type_var, palette=draw_color_palette)
    plt.legend()
    # plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()
