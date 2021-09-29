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


draw_color_palette = {}

def append_to_dataframe(TEST_DATA_DIR, train_type, cost_results_df):
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

    return cost_results_df


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()
    fig_ext = args.fig_ext

    # if you have two pca_full result folders, 
    # rename them as `pca_full_1` and `pca_full_2`
    # similar for `pca_mpc`
    sub_dirs = ['pca_full_1', 'pca_full_2']

    optimal_costs_dict = {}

    cost_results_df = pandas.DataFrame()

    for i in range(len(sub_dirs)):

        sub_dir = sub_dirs[i]
        BASE_DIR = SCRATCH_DIR + sub_dir
        
        cost_var = r'Control Cost $J^\mathrm{c}$'
        fcst_var = r'Forecasting Error $J^\mathrm{F}$ (MSE)'
        ctrl_var = r'Control Error (MSE)'
        size_var = r"Bottleneck Dimension $Z$"
        train_type_var = 'Policy'

        PLOT_DIR = BASE_DIR + '/pca_results/'


        plot_optimal_line = True

        result_dict = load_pkl(PLOT_DIR + '/optimal/optimal.pkl')
        optimal_cost = result_dict['cost_mean']

        optimal_costs_dict[sub_dir] = optimal_cost

        print("optimal_cost is: {}".format(optimal_cost))

        for train_type in ['task_aware_first_control', 'task_agnostic']:

            TEST_DATA_DIR = PLOT_DIR + train_type
            plot_train_type = train_type_variable_to_name[train_type] + ' (Case {})'.format(i+1)
            draw_color_palette[plot_train_type] = train_type_variable_to_color[train_type]
            cost_results_df = append_to_dataframe(TEST_DATA_DIR, train_type, cost_results_df)


    plot_file = PLOT_DIR + '/pca_cost_bottleneck_combined.{}'.format(fig_ext)

    
    sns.pointplot(x=size_var, y=cost_var, data=cost_results_df, hue=train_type_var, 
                  palette=draw_color_palette, markers=["o", "o", "x", "x"], 
                  linestyles=["-", "-", "-.", "-."])
    
    line_styles = ['--', '-.']
    for i in range(len(sub_dirs)):
        sub_dir = sub_dirs[i]
        line_style = line_styles[i]
        if plot_optimal_line:
            plt.axhline(y = optimal_costs_dict[sub_dir], linewidth = 2.0, ls = line_style, 
                        color = 'black', label = 'Optimal'+ ' (Case {})'.format(i+1))

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    by_label = OrderedDict(sorted(by_label.items()))
    plt.legend(by_label.values(), by_label.keys())

    # plt.legend()
    plt.savefig(plot_file)
    plt.close()
