import numpy as np
import os, sys
import pandas
import argparse

ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR'] 
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')

SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/'

from textfile_utils import *
from plotting_utils import *
from collections import OrderedDict


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_type', type=str)
    parser.add_argument('--z_dim', type=int)
    parser.add_argument('--freq', type=int, default=10)
    parser.add_argument('--fig_ext', type=str)
    args = parser.parse_args()

    model_name = args.model_name
    train_type = args.train_type
    z_dim = args.z_dim
    freq = args.freq
    fig_ext = args.fig_ext

    BASE_DIR = SCRATCH_DIR + model_name
    
    train_var = r'Train Loss'
    epoch_var = r'Epoch'

    PLOT_DIR = BASE_DIR + '/test_results/'

    train_loss_results_df = pandas.DataFrame()

    train_type_plot_vars = OrderedDict()


    TEST_DATA_DIR = PLOT_DIR + train_type

    pkl_file = 'z_{}.pkl'.format(z_dim)

      
    result_dict = load_pkl(TEST_DATA_DIR + '/' + pkl_file)

    s_dim = result_dict['s_dim']
    z_dim = result_dict['z_dim']
    train_losses = result_dict['train_losses']
    W = result_dict['W']
    H = result_dict['H']

    # attach to cost_results_df
    basic_results_df = pandas.DataFrame()
    train_loss_results_df[train_var] = train_losses
    train_loss_results_df[epoch_var] = list(range(freq, freq * len(train_losses) + 1, freq))
 

    title_str = 'W = {}, H = {}, s_dim = {}, z_dim = {}'.format(W, H, s_dim, z_dim)

    plot_file = PLOT_DIR + '/{}_{}_train_loss_evoluation.{}'.format(model_name, train_type, fig_ext)
    train_loss_results_df.plot(x=epoch_var, y=train_var)
    plt.legend()
    plt.title(title_str)
    plt.savefig(plot_file)
    plt.close()
