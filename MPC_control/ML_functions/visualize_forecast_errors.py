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
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    z_dims = args.z_dims.split(',')
    fig_ext = args.fig_ext

    train_types = args.train_types.split(',')


    BASE_DIR = SCRATCH_DIR + model_name

    PLOT_DIR = BASE_DIR + '/visualize_forecasts_errors/'
    remove_and_create_dir(PLOT_DIR)

    for subfolder_name in ['forecast_horizon', 'forecast_signal_dimension', 'forecast_heatmap', 'control_signal_dimension']:
        remove_and_create_dir('{}/{}/'.format(PLOT_DIR, subfolder_name))
        for z_dim_str in z_dims: 
            remove_and_create_dir('{}/{}/z_{}/'.format(PLOT_DIR, subfolder_name, z_dim_str))

    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/results.pkl')

    loss_fn = torch.nn.MSELoss()

    fcst_result_dict = OrderedDict()
    ctrl_result_dict = OrderedDict()


    T = data_dict['T']
    s_dim = data_dict['s_dim']
    u_dim = data_dict['u_dim']
    W = data_dict['W']
    H = data_dict['H']

    if "optimal" in train_types:
        train_types.remove("optimal")
    
    for z_dim_str in z_dims:
        z_dim = int(z_dim_str)
        for train_type in train_types:

            RESULT_DIR = BASE_DIR + '/test_results/' + str(train_type)
            result_dict = load_pkl(RESULT_DIR + '/z_' + str(z_dim) + '.pkl')

            fcst_diffs = result_dict['fcst_diffs'] 
            ctrl_diffs = result_dict['ctrl_diffs']

            fcst_diffs_by_dim = [[] for i in range(H * s_dim)]
            ctrl_diffs_by_dim = [[] for i in range(s_dim)]

            for k in range(H * s_dim):
                fcst_diffs_by_dim[k] = fcst_diffs[:,k].tolist()
                        
            for k in range(s_dim):
                ctrl_diffs_by_dim[k] = ctrl_diffs[:,k].tolist() 


            for k in range(H * s_dim):
                fcst_diffs_by_dim[k] = np.array(fcst_diffs_by_dim[k])
            for k in range(s_dim):
                ctrl_diffs_by_dim[k] = np.array(ctrl_diffs_by_dim[k])

            # for MSE loss instead of pure difference, uncomment this piece of code
            for k in range(H * s_dim):
                fcst_diffs_by_dim[k] = fcst_diffs_by_dim[k] ** 2
            for k in range(s_dim):
                ctrl_diffs_by_dim[k] = ctrl_diffs_by_dim[k] ** 2 

            fcst_result_dict[train_type] = fcst_diffs_by_dim
            ctrl_result_dict[train_type] = ctrl_diffs_by_dim

        fcst_err = r"Forecasting Error $J^\mathrm{F}$ (MSE)"
        ctrl_err = r"Control Error (MSE)"
        full_s_idx = r"Index in full_s"
        s_idx = r"Timeseries Element $s(i)$"
        u_idx = r"Timeseries Element $u(i)$"
        t_idx = r"Relative Time Index after $t$"
        train_type_var = r"Policy"

        # # Fig I: no aggregation
        # fcst_err_df = pandas.DataFrame()

        # for train_type in train_types:
        #     if train_type == 'task_aware_first_control':
        #         continue
        #     for k in range(H * s_dim):
        #         basic_results_df = pandas.DataFrame()
        #         basic_results_df[fcst_err] = fcst_result_dict[train_type][k]
        #         basic_results_df[full_s_idx] = [ k ] * len(fcst_result_dict[train_type][k])
        #         basic_results_df[train_type_var] = [ train_type_variable_to_name[train_type] ] * len(fcst_result_dict[train_type][k])

        #         fcst_err_df = fcst_err_df.append(basic_results_df)

        # print("df number of rows: {}".format(fcst_err_df.shape[0]))
        # plot_file = PLOT_DIR  + 'forecasts_errors' + '_z_dim_{}'.format(z_dim) + '.{}'.format(fig_ext)
        # sns.pointplot(x=full_s_idx, y=fcst_err, data=fcst_err_df, hue=train_type_var, palette=draw_color_palette)

        # plt.legend()
        # plt.savefig(plot_file)
        # plt.close()

        # Fig I(2): heatmap
        fcst_err_dfs = OrderedDict()
        for train_type in train_types:
            if train_type == 'task_aware_first_control':
                continue

            fcst_err_df = pandas.DataFrame()
            for k in range(H * s_dim):
                curr_s_idx = k % s_dim
                curr_t_idx = int (k / s_dim)

                basic_results_df = pandas.DataFrame()
                basic_results_df[s_idx] = [ curr_s_idx ]
                basic_results_df[t_idx] = [ curr_t_idx ]
                basic_results_df[train_type_var] = [ np.mean(fcst_result_dict[train_type][k]) ]

                # print(basic_results_df)
                fcst_err_df = fcst_err_df.append(basic_results_df)

            fcst_err_df = fcst_err_df.pivot(s_idx, t_idx, train_type_var)
            fcst_err_df.index.name = None
            fcst_err_df.columns.name = None

            fcst_err_dfs[train_type] = fcst_err_df
            print(fcst_err_df)

            print("df number of rows: {}".format(fcst_err_df.shape[0]))

        plot_file = '{}/forecast_heatmap/z_{}/'.format(PLOT_DIR, z_dim_str) + \
                    'forecast_errors.{}'.format(fig_ext)

        selected_train_types = list(fcst_err_dfs.keys())
        v_min = 0
        v_max = 0
        for train_type in selected_train_types:
            v_max = max(v_max, fcst_err_dfs[train_type].to_numpy().max())
        print(v_max)

        print(selected_train_types)
        fig, axn = plt.subplots(len(selected_train_types), 1, sharex=True, sharey=True)
        for i in range(len(selected_train_types)):
        # for i, ax in enumerate(axn.flat):
            ax = axn[len(selected_train_types)-1-i]
            train_type = selected_train_types[i-1]
            fcst_err_df = fcst_err_dfs[train_type]
            sns.heatmap(fcst_err_df, ax=ax, vmin=v_min, vmax=v_max, cmap="YlGnBu")
            ax.set_title(train_type_variable_to_name[train_type])
            # sns.heatmap(fcst_err_df, ax=ax, cmap="YlGnBu")
            # icml figures shouldn't have title
            # plt.title(fcst_err) 

        # the alignment is strange, so put some blank spaces in the beginning
        # plt.ylabel("                    " + s_idx)
        plt.ylabel("                        " + s_idx)
        plt.xlabel(t_idx)

        plt.savefig(plot_file)
        plt.close()


        # # Fig II: aggregate by s_dim
        # fcst_err_df = pandas.DataFrame()

        # for train_type in train_types:
        #     if train_type == 'task_aware_first_control':
        #         continue
        #     for k in range(H * s_dim):
        #         dim_in_s_dim = k % s_dim
        #         basic_results_df = pandas.DataFrame()
        #         basic_results_df[fcst_err] = fcst_result_dict[train_type][k]
        #         basic_results_df[s_idx] = [ dim_in_s_dim ] * len(fcst_result_dict[train_type][k])
        #         basic_results_df[train_type_var] = [ train_type_variable_to_name[train_type] ] * len(fcst_result_dict[train_type][k])

        #         fcst_err_df = fcst_err_df.append(basic_results_df)

        # print("df number of rows: {}".format(fcst_err_df.shape[0]))
        # plot_file = PLOT_DIR  + 'forecasts_errors' + '_z_dim_{}_aggregated_by_s_dim'.format(z_dim) + '.{}'.format(fig_ext)
        # sns.pointplot(x=s_idx, y=fcst_err, data=fcst_err_df, hue=train_type_var, palette=draw_color_palette)

        # plt.legend()
        # plt.savefig(plot_file)
        # plt.close()

        # Fig II(2): only present s_dim for current time interval
        fcst_err_df = pandas.DataFrame()

        for train_type in train_types:
            if train_type == 'task_aware_first_control':
                continue
            for k in range(s_dim):
                basic_results_df = pandas.DataFrame()
                basic_results_df[fcst_err] = fcst_result_dict[train_type][k]
                basic_results_df[s_idx] = [ k ] * len(fcst_result_dict[train_type][k])
                basic_results_df[train_type_var] = [ train_type_variable_to_name[train_type] ] * len(fcst_result_dict[train_type][k])

                fcst_err_df = fcst_err_df.append(basic_results_df)

        print("df number of rows: {}".format(fcst_err_df.shape[0]))
        plot_file = '{}/forecast_signal_dimension/z_{}/'.format(PLOT_DIR, z_dim_str) + \
                    'current_forecast_errors_signal_dimension.{}'.format(fig_ext)
        sns.pointplot(x=s_idx, y=fcst_err, data=fcst_err_df, hue=train_type_var, palette=draw_color_palette)

        plt.legend()
        plt.savefig(plot_file)
        plt.close()


        # Fig III: aggregate by time
        fcst_err_df = pandas.DataFrame()

        for train_type in train_types:
            if train_type == 'task_aware_first_control':
                continue
            for k in range(H * s_dim):
                time_idx = int(k / s_dim)
                basic_results_df = pandas.DataFrame()
                basic_results_df[fcst_err] = fcst_result_dict[train_type][k]
                basic_results_df[t_idx] = [ time_idx ] * len(fcst_result_dict[train_type][k])
                basic_results_df[train_type_var] = [ train_type_variable_to_name[train_type] ] * len(fcst_result_dict[train_type][k])

                fcst_err_df = fcst_err_df.append(basic_results_df)

        print("df number of rows: {}".format(fcst_err_df.shape[0]))
        plot_file = '{}/forecast_horizon/z_{}/'.format(PLOT_DIR, z_dim_str) + \
                    'forecast_errors_time_horizon.{}'.format(fig_ext)
        sns.pointplot(x=t_idx, y=fcst_err, data=fcst_err_df, hue=train_type_var, palette=draw_color_palette)

        plt.legend()
        plt.savefig(plot_file)
        plt.close()


        # Fig IV: control diffs
        ctrl_err_df = pandas.DataFrame()

        for train_type in train_types:
            for k in range(u_dim):
                basic_results_df = pandas.DataFrame()
                basic_results_df[ctrl_err] = ctrl_result_dict[train_type][k]
                basic_results_df[u_idx] = [ k ] * len(ctrl_result_dict[train_type][k])
                basic_results_df[train_type_var] = [ train_type_variable_to_name[train_type] ] * len(ctrl_result_dict[train_type][k])

                ctrl_err_df = ctrl_err_df.append(basic_results_df)

        print("df number of rows: {}".format(ctrl_err_df.shape[0]))
        plot_file = '{}/control_signal_dimension/z_{}/'.format(PLOT_DIR, z_dim_str) + \
                    'control_errors.{}'.format(fig_ext)
        sns.pointplot(x=u_idx, y=ctrl_err, data=ctrl_err_df, hue=train_type_var, palette=draw_color_palette)

        plt.legend()
        plt.savefig(plot_file)
        plt.close()
