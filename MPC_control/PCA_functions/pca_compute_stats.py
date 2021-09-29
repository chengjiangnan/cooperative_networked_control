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
from utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--train_types', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--allowed_error_per', type=int, default=5)
    args = parser.parse_args()
    model_name = args.model_name
    train_types = args.train_types.split(',')
    allowed_error_per = args.allowed_error_per

    BASE_DIR = SCRATCH_DIR + model_name
    PLOT_DIR = BASE_DIR + '/pca_results/'

    avr_cost = {}
    min_z_dims = {}
    opt_costs = {}
    best_z_dims = {}


    plot_optimal_line = False
    if "optimal" in train_types:
        train_types.remove("optimal")

        result_dict = load_pkl(PLOT_DIR + '/optimal/optimal.pkl')
        optimal_cost = result_dict['cost_mean']

        avr_cost["optimal"] = optimal_cost

        print("optimal_cost is: {}".format(optimal_cost))

    for train_type in train_types:

        if train_type == 'weighted':
            continue

        avr_cost[train_type] = {}

        # the minimum z_dim to achieve a cost close to the optimal cost
        min_z_dim = 10000

        TEST_DATA_DIR = PLOT_DIR + train_type

        pkl_file_list = [x for x in os.listdir(TEST_DATA_DIR) if '.pkl' in x]

        opt_cost = 10000
        best_z_dim = -1

        for pkl_file in pkl_file_list:      
            result_dict = load_pkl(TEST_DATA_DIR + '/' + pkl_file)
            s_dim = result_dict['s_dim']
            z_dim = result_dict['z_dim']
            H = result_dict['H']
            cost = result_dict['cost_mean']
            avr_cost[train_type][z_dim] = cost

            full_dim = s_dim * H
            if cost <= optimal_cost * (1 + allowed_error_per / 100):
                min_z_dim = min(min_z_dim, z_dim)

            if cost <= opt_cost:
                opt_cost = cost
                best_z_dim = z_dim

            print("[{}] z_dim: {}, compression gain: {}, "
                  "cost/optimal_cost: {}"
                  .format(train_type, z_dim, full_dim / z_dim, cost/optimal_cost))

        min_z_dims[train_type] = min_z_dim
        opt_costs[train_type] = opt_cost
        best_z_dims[train_type] = best_z_dim



    full_dim = s_dim * H
    task_agnostic_z_dim = min_z_dims["task_agnostic"]

    print("--------------")

    for train_type in train_types:

        if train_type == 'weighted':
            continue

        min_z_dim = min_z_dims[train_type]
        compression_gain = full_dim / min_z_dim
        policy_gain = task_agnostic_z_dim / min_z_dim

        print("[{}] min_zdim: {}, compression gain: {}, policy gain: {}"
              .format(train_type, min_z_dim, compression_gain, policy_gain))

        print("[{}] opt_cost/optimal_cost: {}, best_z_dim: {}"
              .format(train_type, opt_costs[train_type]/optimal_cost,
                      best_z_dims[train_type]))

    print("--------------")

    print(avr_cost)
