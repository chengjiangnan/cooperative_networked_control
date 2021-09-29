import torch
import torch.nn as nn

torch.manual_seed(0)

from MPC_forecaster.forecaster import *
from MPC_controller.simple_MPC_controller import *
from MPC_controller.box_MPC_controller import *
from MPC_controller.biased_MPC_controller import *

import seaborn as sns

def init_controller(controller_name, data_dict):
    if controller_name == "simple_mpc":
        mpc_paras = SimpleMpcParas(
            data_dict['x_dim'], data_dict['u_dim'], data_dict['s_dim'], 
            data_dict['A'], data_dict['B'], data_dict['C'], data_dict['Q'], data_dict['R'], data_dict['H'])
        controller = SimpleMpcController(mpc_paras, device)
    elif controller_name == "box_mpc":
        mpc_paras = BoxMpcParas(
            data_dict['x_dim'], data_dict['u_dim'], data_dict['s_dim'],
            data_dict['u_ub'], data_dict['u_lb'],
            data_dict['A'], data_dict['B'], data_dict['C'], data_dict['Q'], data_dict['R'], data_dict['H'])
        controller = BoxMpcController(mpc_paras, device)
    elif controller_name == "biased_mpc":
        mpc_paras = BiasedMpcParas(
            data_dict['x_dim'], data_dict['u_dim'], data_dict['s_dim'],
            data_dict['A'], data_dict['B'], data_dict['C'], 
            data_dict['Q_h'], data_dict['Q_b'], data_dict['R'], data_dict['H'])
        controller = BiasedMpcController(mpc_paras, device)
    else:
        raise Exception("undefined controller_name: {}".format(controller_name))

    return controller

def init_forecaster(forecaster_name, model_params):
    if forecaster_name == "DNN_forecaster":
        forecaster = DNNForecaster(model_params).to(device)
    # elif forecaster_name == "DNN3_forecaster":
    #     forecaster = DNN3Forecaster(model_params).to(device)
    elif forecaster_name =="simple_forecaster":
        forecaster = SimpleForecaster(model_params).to(device)
    elif forecaster_name =="LSTM_forecaster":
        forecaster = LSTMForecaster(model_params, device).to(device)
    else:
        raise Exception("undefined forecaster_name: {}".format(forecaster_name))
    return forecaster

train_type_variable_to_name = {
                                'task_agnostic': 'Task-agnostic',
                                'task_aware_first_control': 'Task-aware' + r', $\lambda^\mathrm{F}$=0',
                                'weighted': 'Weighted',
                                # 'weighted': 'Weighted' + r', $\lambda^\mathrm{F}$=1.0',
                                'task_aware_MPC_cost': 'Task Aware (MPC cost)',
                                'optimal': 'Optimal'
                              }

palette_colors = sns.color_palette()
train_type_variable_to_color = {
                                'task_agnostic': palette_colors[0],
                                'task_aware_first_control': palette_colors[1],
                                'weighted': palette_colors[2],
                                'task_aware_MPC_cost': palette_colors[3],
                                'optimal': 'black'
                               }
draw_color_palette = {}

for train_type in train_type_variable_to_name.keys():
    draw_color_palette[train_type_variable_to_name[train_type]] = train_type_variable_to_color[train_type]
