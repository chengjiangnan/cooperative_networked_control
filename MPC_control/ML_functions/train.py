"""
    train an distributed estimator with a fixed controller
"""
import sys, os
import torch
import torch.nn as nn
import argparse

torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROBOTICS_CODESIGN_DIR = os.environ['ROBOTICS_CODESIGN_DIR'] 
sys.path.append(ROBOTICS_CODESIGN_DIR)
sys.path.append(ROBOTICS_CODESIGN_DIR + '/utils/')
sys.path.append(ROBOTICS_CODESIGN_DIR + '/MPC_control/')

SCRATCH_DIR = ROBOTICS_CODESIGN_DIR + '/scratch/'

from utils import *

from textfile_utils import *
from plotting_utils import *

from collections import OrderedDict


def train_integrated_forecaster_MPC(forecaster, controller, train_options, data_dict):

    # setup optimizer
    optimizer = torch.optim.Adam(
        forecaster.parameters(), lr=train_options["learning_rate"], amsgrad=True)

    loss_fn = torch.nn.MSELoss().to(device)

    train_losses = []
    # loop over epochs
    for i in range(train_options["num_epochs"]):
        s_past = data_dict['train_inputs'].to(device)
        s_future = data_dict['train_outputs'].to(device)

        s_past_2d = s_past.reshape(s_past.shape[0], s_past.shape[1])
        s_future_2d = s_future.reshape(s_future.shape[0], s_future.shape[1])
        s_future_hat_2d = forecaster(s_past_2d)

        if train_options["train_type"] == 'task_agnostic':
            train_loss = loss_fn(s_future_hat_2d, s_future_2d)
        elif train_options["train_type"] == 'task_aware_first_control':
            x0 = torch.zeros([s_past.shape[0], data_dict['x_dim'], 1], dtype=torch.float32).to(device)
            x_sol_star, u_sol_star = controller.forward(
                x0, s_future.reshape(s_future.shape[0], data_dict['H'], data_dict['s_dim']))
            x_sol_hat, u_sol_hat = controller.forward(
                x0, s_future_hat_2d.reshape(s_future_hat_2d.shape[0], data_dict['H'], data_dict['s_dim']))
            train_loss = loss_fn(u_sol_hat[:, 0:1, :], u_sol_star[:, 0:1, :])
        elif train_options["train_type"] == 'task_aware_MPC_cost':
            x0 = torch.zeros([s_past.shape[0], data_dict['x_dim'], 1], dtype=torch.float32).to(device)
            x_sol_star, u_sol_star = controller.forward(
                x0, s_future.reshape(s_future.shape[0], data_dict['H'], data_dict['s_dim']))
            x_sol_hat, u_sol_hat = controller.forward(
                x0, s_future_hat_2d.reshape(s_future_hat_2d.shape[0], data_dict['H'], data_dict['s_dim']))
            train_loss = controller.task_loss(u_sol_hat, u_sol_star)
        elif train_options["train_type"] == 'weighted':
            w = 0.5
            x0 = torch.zeros([s_past.shape[0], data_dict['x_dim'], 1], dtype=torch.float32).to(device)
            x_sol_star, u_sol_star = controller.forward(
                x0, s_future.reshape(s_future.shape[0], data_dict['H'], data_dict['s_dim']))
            x_sol_hat, u_sol_hat = controller.forward(
                x0, s_future_hat_2d.reshape(s_future_hat_2d.shape[0], data_dict['H'], data_dict['s_dim']))

            # first part based on 'task_aware_first_control'
            train_loss_1 = loss_fn(u_sol_hat[:, 0:1, :], u_sol_star[:, 0:1, :])
            
            # # first part based on 'task_aware_MPC_cost'
            # train_loss_1 = controller.task_loss(u_sol_hat, u_sol_star)

            train_loss_2 = loss_fn(s_future_hat_2d, s_future_2d)
            train_loss = w * train_loss_1 + (1-w) * train_loss_2
        else:
            raise Exception("undefined train_type: {}".format(train_options["train_type"]))

        # take a step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if (i + 1) % train_options["output_freq"] == 0:
            print(" ")
            print("Epoch %d: Train RMSE %0.3f" % (i + 1, train_loss) + ' ' + train_options["train_type"])
            train_losses.append(train_loss.item())


    if train_options["output_recon_loss"] == True:
        forecaster.eval()

        recon_loss = 0

        s_past = data_dict['train_inputs'].to(device)
        s_future = data_dict['train_outputs'].to(device)

        s_past_2d = s_past.reshape(s_past.shape[0], s_past.shape[1])
        s_future_2d = s_future.reshape(s_future.shape[0], s_future.shape[1])
        s_future_hat_2d = forecaster(s_past_2d)

        recon_loss = loss_fn(s_future_hat_2d, s_future_2d)

        return train_losses, recon_loss.item()

    return train_losses, None


def one_control_step(controller, x0, s0, s_future_3d):

    x_sol, u_sol = controller.forward(x0, s_future_3d)
    u0 = u_sol[:, 0:1, :].transpose(1, 2)
    x1 = controller.next_state(x0, u0, s0)
    one_step_cost = controller.one_step_cost(u0, x1)

    return x1, u0, one_step_cost


def evaluate_forecaster_cost(forecaster, controller, data_dict):

    # val_dataset: (batch_size, T*s_dim, 1)
    val_dataset = data_dict['val_dataset'][:, :, :]
    val_dataset = val_dataset.to(device)

    x0 = torch.zeros([val_dataset.shape[0], data_dict['x_dim'], 1], dtype=torch.float32).to(device)

    cost = torch.zeros(val_dataset.shape[0], 1, 1).to(device)

    num_samples = val_dataset.shape[0]
    x_array = np.zeros((num_samples, data_dict['T']-data_dict['H']-data_dict['W']+3, data_dict['x_dim']))
    u_array = np.zeros((num_samples, data_dict['T']-data_dict['H']-data_dict['W']+2, data_dict['u_dim']))

    for t in range(data_dict['W']-1, data_dict['T']-data_dict['H']+1):

        s_past_3d = val_dataset[:, (t-data_dict['W']+1)*data_dict['s_dim']:(t+1)*data_dict['s_dim'], :]
        s_past_2d = s_past_3d.reshape(s_past_3d.shape[0], s_past_3d.shape[1])
        s_future_hat_2d = forecaster(s_past_2d)
        s_future_hat_3d = s_future_hat_2d.reshape(s_future_hat_2d.shape[0], data_dict['H'], data_dict['s_dim'])
        s0 = val_dataset[:, t*data_dict['s_dim']:(t+1)*data_dict['s_dim'], :]

        x1, u0, one_step_cost = one_control_step(controller, x0, s0, s_future_hat_3d)
        
        cost += one_step_cost
        x0 = x1

        t_idx = t - (data_dict['W']-1)
        x_array[ :, t_idx+1, : ] = x1.reshape(x1.shape[0], x1.shape[1]).cpu().detach().numpy()
        u_array[ :, t_idx, :] = u0.reshape(u0.shape[0], u0.shape[1]).cpu().detach().numpy()


    cost_array = cost.reshape(cost.shape[0]).cpu().detach().numpy()

    return cost_array, x_array, u_array


def evaluate_forecaster_diffs(forecaster, controller, data_dict):

    s_past = data_dict['val_inputs'].to(device)
    s_future = data_dict['val_outputs'].to(device)

    s_past_2d = s_past.reshape(s_past.shape[0], s_past.shape[1])
    s_future_2d = s_future.reshape(s_future.shape[0], s_future.shape[1])
    s_future_hat_2d = forecaster(s_past_2d)

    x0 = torch.zeros([s_past.shape[0], data_dict['x_dim'], 1], dtype=torch.float32).to(device)
    x_sol_star, u_sol_star = controller.forward(
        x0, s_future.reshape(s_future.shape[0], data_dict['H'], data_dict['s_dim']))
    x_sol_hat, u_sol_hat = controller.forward(
        x0, s_future_hat_2d.reshape(s_future_hat_2d.shape[0], data_dict['H'], data_dict['s_dim']))

    fcst_diffs = s_future_hat_2d - s_future_2d
    ctrl_diffs = u_sol_hat[:, 0, :] - u_sol_star[:, 0, :]

    return fcst_diffs, ctrl_diffs


def optimal_cost(controller, data_dict):

    # val_dataset: (batch_size, T*s_dim, 1)
    val_dataset = data_dict['val_dataset'][:, :, :]
    val_dataset = val_dataset.to(device)

    x0 = torch.zeros([val_dataset.shape[0], data_dict['x_dim'], 1], dtype=torch.float32).to(device)

    cost = torch.zeros(val_dataset.shape[0], 1, 1).to(device)

    num_samples = val_dataset.shape[0]
    x_array = np.zeros((num_samples, data_dict['T']-data_dict['H']-data_dict['W']+3, data_dict['x_dim']))
    u_array = np.zeros((num_samples, data_dict['T']-data_dict['H']-data_dict['W']+2, data_dict['u_dim']))

    for t in range(data_dict['W']-1, data_dict['T']-data_dict['H']+1):
        s_future_3d = val_dataset[:, t*data_dict['s_dim']:(t+data_dict['H'])*data_dict['s_dim'], :]
        s_future_3d.reshape(s_future_3d.shape[0], data_dict['H'], data_dict['s_dim'])
        s0 = val_dataset[:, t*data_dict['s_dim']:(t+1)*data_dict['s_dim'], :]

        x1, u0, one_step_cost = one_control_step(controller, x0, s0, s_future_3d)
        
        cost += one_step_cost
        x0 = x1

        t_idx = t - (data_dict['W']-1)
        x_array[ :, t_idx+1, : ] = x1.reshape(x1.shape[0], x1.shape[1]).cpu().detach().numpy()
        u_array[ :, t_idx, :] = u0.reshape(u0.shape[0], u0.shape[1]).cpu().detach().numpy()

    cost_array = cost.reshape(cost.shape[0]).cpu().detach().numpy()

    return cost_array, x_array, u_array



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--controller_name', type=str)
    parser.add_argument('--forecaster_name', type=str)
    parser.add_argument('--forecaster_hidden_dim', type=int)
    parser.add_argument('--train_types', type=str)
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--min_z', type=int, default=1)
    parser.add_argument('--max_z', type=int, default=9)
    args = parser.parse_args()

    model_name = args.model_name
    controller_name = args.controller_name
    forecaster_name = args.forecaster_name
    forecaster_hidden_dim = args.forecaster_hidden_dim
    num_epochs = args.num_epochs
    train_types = args.train_types.split(',')
    min_z = args.min_z
    max_z = args.max_z

    BASE_DIR = SCRATCH_DIR + model_name
    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/results.pkl')

    controller_train = init_controller(controller_name, data_dict)
    controller_val = init_controller(controller_name, data_dict)

    # params for forecaster model
    model_params = OrderedDict()
    model_params["W"] = data_dict['W']
    model_params["H"] = data_dict['H']
    model_params["s_dim"] = data_dict['s_dim']
    model_params["hidden_dim"] = forecaster_hidden_dim

    train_options = { 
                      "num_epochs": num_epochs,
                      "learning_rate": 1e-3,
                      "output_freq": 10,
                      "output_recon_loss": True,
                      "save_model": True
                    }

    for train_type in train_types:
    # for train_type in ['task_aware_supervised', 'task_agnostic_RMSE']:
    # for train_type in ['task_agnostic_RMSE']:
        TEST_DATA_DIR = BASE_DIR + '/test_results/' + train_type
        remove_and_create_dir(TEST_DATA_DIR)

        if train_type == "optimal":
            cost_array, x_array, u_array = optimal_cost(controller_val, data_dict)
            
            result_dict = OrderedDict()
            result_dict['cost_mean'] = np.mean(cost_array)
            result_dict['cost_std'] = np.std(cost_array)
            result_dict['cost_array'] = cost_array
            result_dict['x_array'] = x_array
            result_dict['u_array'] = u_array

            print("Average cost: {}".format(result_dict['cost_mean']))

            write_pkl(fname = TEST_DATA_DIR + '/optimal.pkl', input_dict = result_dict)

            continue

        MODEL_DIR = BASE_DIR + '/trained_models/' + train_type
        remove_and_create_dir(MODEL_DIR)
        
        results_file = TEST_DATA_DIR + '/overall.txt'

        train_options["train_type"] = train_type
        
        with open(results_file, 'w') as f:
            header_str = '\t'.join(
                ['train_type', 'z', 'mean_cost_J', 'std_cost_J', 'num_epochs', 'train_loss', 'recon_loss'])
            f.write(header_str + '\n')

            for z_dim in range(min_z, max_z + 1):
                print('################')
                print('latent_dim: ', z_dim)

                model_params["z_dim"] =  z_dim

                # set up forecasting model
                forecaster = init_forecaster(forecaster_name, model_params)

                train_losses, recon_loss = train_integrated_forecaster_MPC(
                    forecaster, controller_train, train_options, data_dict)
                train_loss = train_losses[-1]
                forecaster.eval()

                cost_array, x_array, u_array = evaluate_forecaster_cost(forecaster, controller_val, data_dict)
                fcst_diffs, ctrl_diffs = evaluate_forecaster_diffs(forecaster, controller_val, data_dict)

                # save for plotting later
                result_dict = OrderedDict()
                result_dict['train_losses'] = train_losses
                result_dict['train_loss'] = train_loss
                result_dict['recon_loss'] = recon_loss
                result_dict['cost_mean'] = np.mean(cost_array)
                result_dict['cost_std'] = np.std(cost_array)
                result_dict['cost_array'] = cost_array
                result_dict['x_array'] = x_array
                result_dict['u_array'] = u_array
                result_dict['fcst_diffs'] = fcst_diffs
                result_dict['ctrl_diffs'] = ctrl_diffs
                result_dict['s_dim'] = data_dict['s_dim']
                result_dict['z_dim'] = z_dim
                result_dict['W'] = data_dict['W']
                result_dict['H'] = data_dict['H']

                print("Average cost: {}".format(result_dict['cost_mean']))

                write_pkl(fname = TEST_DATA_DIR + '/z_{}.pkl'.format(z_dim), input_dict = result_dict)
                
                out_str = '\t'.join([ train_type, str(z_dim), str(result_dict['cost_mean']), 
                                      str(result_dict['cost_std']), str(num_epochs), 
                                      str(train_loss), str(recon_loss)])
                f.write(out_str + '\n')

                if train_options["save_model"]:
                    model_path = MODEL_DIR + '/z_{}.pt'.format(z_dim)

                    model_save_dict = { "num_epochs": num_epochs,
                                        "forecaster_state_dict": forecaster.state_dict()
                                      }
                    torch.save(model_save_dict, model_path)
