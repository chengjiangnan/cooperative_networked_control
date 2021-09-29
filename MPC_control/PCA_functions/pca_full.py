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

from numpy.linalg import eig

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def one_control_step_for_known_control(controller, x0, s0, u0):
    x1 = controller.next_state(x0, u0, s0)
    one_step_cost = controller.one_step_cost(u0, x1)

    return x1, one_step_cost


def full_control_steps(controller, s_future, s_future_hat, data_dict):
    num_samples = s_future.shape[0]
    cost = torch.zeros(num_samples, 1, 1).to(device)
    x_array = np.zeros((num_samples, data_dict['T']+1, data_dict['x_dim']))
    u_array = np.zeros((num_samples, data_dict['T'], data_dict['u_dim']))

    x0 = torch.zeros([num_samples, data_dict['x_dim'], 1], dtype=torch.float32).to(device)

    _, u_sol = controller.forward(x0, s_future_hat)

    for t in range(data_dict['T']):
        s0 = s_future[:, t*data_dict['s_dim']:(t+1)*data_dict['s_dim'], :]
        u0 = u_sol[:, t:(t+1), :].transpose(1, 2)
        x1, one_step_cost = one_control_step_for_known_control(controller, x0, s0, u0)
        cost += one_step_cost
        x0 = x1

        x_array[ :, t+1, : ] = x1.reshape(x1.shape[0], x1.shape[1]).cpu().detach().numpy()
        u_array[ :, t, :] = u0.reshape(u0.shape[0], u0.shape[1]).cpu().detach().numpy()

    return cost, x_array, u_array

def evaluate_forecaster_cost(projection, pca, std_scaler, controller, data_dict, train_type):

    # val_dataset: (batch_size, T*s_dim, 1)
    val_dataset = data_dict['val_dataset'][:, :, :]
    val_dataset = val_dataset.to(device)

    num_samples = val_dataset.shape[0]

    s_future = val_dataset

    pca_X = torch.matmul(projection, s_future)
    pca_X = pca_X.reshape(pca_X.shape[0], pca_X.shape[1]).numpy() 

    scaled_pca_X = std_scaler.transform(pca_X)
    pca_X_encode = pca.transform(scaled_pca_X)
    pca_X_decode = pca.inverse_transform(pca_X_encode)
    pca_X_hat = std_scaler.inverse_transform(pca_X_decode)

    pca_X_hat = torch.Tensor(pca_X_hat)
    pca_X_hat = pca_X_hat.reshape(pca_X_hat.shape[0], pca_X_hat.shape[1], 1)

    pca_X_hat_copy = pca_X_hat

    if train_type == "task_aware_first_control":
        pca_X_hat = pca_X_hat.reshape(pca_X_hat.shape[0], pca_X_hat.shape[1]).transpose(0, 1).numpy()
        s_future_hat = np.linalg.lstsq(projection.numpy(), pca_X_hat, rcond=None)[0]
        s_future_hat = torch.Tensor(s_future_hat)
        s_future_hat = s_future_hat.transpose(0, 1)
        s_future_hat = s_future_hat.reshape(s_future_hat.shape[0], s_future_hat.shape[1], 1)
    else:
        s_future_hat = torch.matmul(torch.inverse(projection), pca_X_hat)

    loss_fn = torch.nn.MSELoss()

    pca_MSE = loss_fn(torch.matmul(projection, s_future_hat), pca_X_hat_copy).item()
    print("check! pca_MSE: {}".format(pca_MSE))

    cost, x_array, u_array = full_control_steps(controller, s_future, s_future_hat, data_dict)
    

    cost_array = cost.reshape(cost.shape[0]).cpu().detach().numpy()

    return cost_array, x_array, u_array


def evaluate_forecaster_diffs(projection, pca, std_scaler, controller, data_dict, train_type):

    s_future = data_dict['val_dataset'].to(device)

    pca_X = torch.matmul(projection, s_future)
    pca_X = pca_X.reshape(pca_X.shape[0], pca_X.shape[1]).numpy() 

    scaled_pca_X = std_scaler.transform(pca_X)
    pca_X_encode = pca.transform(scaled_pca_X)
    pca_X_decode = pca.inverse_transform(pca_X_encode)
    pca_X_hat = std_scaler.inverse_transform(pca_X_decode)

    pca_X_hat = torch.Tensor(pca_X_hat)
    pca_X_hat = pca_X_hat.reshape(pca_X_hat.shape[0], pca_X_hat.shape[1], 1)

    if train_type == "task_aware_first_control":
        pca_X_hat = pca_X_hat.reshape(pca_X_hat.shape[0], pca_X_hat.shape[1]).transpose(0, 1).numpy()
        s_future_hat = np.linalg.lstsq(projection.numpy(), pca_X_hat, rcond=None)[0]
        s_future_hat = torch.Tensor(s_future_hat)
        s_future_hat = s_future_hat.transpose(0, 1)
        s_future_hat = s_future_hat.reshape(s_future_hat.shape[0], s_future_hat.shape[1], 1)
    else:
        s_future_hat = torch.matmul(torch.inverse(projection), pca_X_hat)

    controller_W_full = controller_val.W_full()

    u = torch.matmul(controller_W_full, s_future)
    u_hat = torch.matmul(controller_W_full, s_future_hat)

    loss_fn = torch.nn.MSELoss()

    fcst_MSE = loss_fn(s_future_hat, s_future).item()
    ctrl_MSE = loss_fn(u_hat, u).item()

    return fcst_MSE, ctrl_MSE


def optimal_cost(controller, data_dict):

    # val_dataset: (batch_size, T*s_dim, 1)
    val_dataset = data_dict['val_dataset'][:, :, :]
    val_dataset = val_dataset.to(device)

    s_future = val_dataset
    cost, x_array, u_array = full_control_steps(controller, s_future, s_future, data_dict)

    cost_array = cost.reshape(cost.shape[0]).cpu().detach().numpy()

    return cost_array, x_array, u_array


def compute_projection(H, s_dim, controller_Psi, train_type, weight=0):
    # for each train type, need to determine the projection of s
    if train_type == 'task_agnostic':
        projection = np.identity(H * s_dim)
    elif train_type == 'weighted':
        A = controller_Psi.numpy()
        A = A + weight * np.identity(H * s_dim)
        eigen_values, eigen_vectors = eig(A)
        
        eigen_values_root = np.zeros(H * s_dim)
        for i in range(H * s_dim):
            if eigen_values[i] > 0:
                eigen_values_root[i] = eigen_values[i] ** 0.5

        projection = np.diag(eigen_values_root).dot(eigen_vectors.transpose())
    elif train_type == 'task_aware_first_control':
        A = controller_Psi.numpy()
        eigen_values, eigen_vectors = eig(A)
        
        selected_eigen_values_root = np.zeros(H * s_dim)
        selected_eigen_vectors = np.zeros((H * s_dim, H * s_dim))
        cnt = 0
        for i in range(H * s_dim):
            if eigen_values[i] > 0:
                selected_eigen_values_root[cnt] = eigen_values[i] ** 0.5
                selected_eigen_vectors[:, cnt] = eigen_vectors[:, i]
                cnt += 1

        selected_eigen_values_root = selected_eigen_values_root[:cnt]
        selected_eigen_vectors = selected_eigen_vectors[:, :cnt]

        projection = np.diag(selected_eigen_values_root).dot(selected_eigen_vectors.transpose())
    else:
        raise Exception("undefined train_type: {}".format(train_type))


    projection = torch.Tensor(projection)

    return projection


def compute_and_save_result(min_z, max_z, projection, data_dict, train_type, TEST_DATA_DIR):
    # train_outputs: (batch_size, H * s_dim, 1)
    train_outputs = data_dict['train_dataset']

    for z_dim in range(min_z, max_z + 1):
            print('################')
            print('latent_dim: ', z_dim)

            if z_dim > projection.shape[0]:
                break

            # pca_X: (batch_size, H * s_dim, 1) for task-agnostic/weighted
            # or: (batch_size, u_dim, 1) for task-aware
            pca_X = torch.matmul(projection, train_outputs)
            pca_X = pca_X.reshape(pca_X.shape[0], pca_X.shape[1]).numpy() 

            # input of pca is X: (n_features, n_dimensions)
            std_scaler = StandardScaler(with_std=False)
            scaled_pca_X = std_scaler.fit_transform(pca_X)
            pca = PCA(n_components=z_dim)
            pca_X_encode = pca.fit_transform(scaled_pca_X)
            pca_X_decode = pca.inverse_transform(pca_X_encode)
            pca_X_hat = std_scaler.inverse_transform(pca_X_decode)

            cost_array, x_array, u_array = evaluate_forecaster_cost(
                projection, pca, std_scaler, controller_val, data_dict, train_type)

            fcst_diff, ctrl_diff = evaluate_forecaster_diffs(
                projection, pca, std_scaler, controller_val, data_dict, train_type)

            # save for plotting later
            result_dict = OrderedDict()
            result_dict['cost_mean'] = np.mean(cost_array)
            result_dict['cost_std'] = np.std(cost_array)
            result_dict['cost_array'] = cost_array
            result_dict['x_array'] = x_array
            result_dict['u_array'] = u_array
            result_dict['fcst_diff'] = fcst_diff
            result_dict['ctrl_diff'] = ctrl_diff
            result_dict['s_dim'] = data_dict['s_dim']
            result_dict['z_dim'] = z_dim
            result_dict['H'] = data_dict['H']

            print("Average cost: {}".format(result_dict['cost_mean']))

            write_pkl(fname = TEST_DATA_DIR + '/z_{}.pkl'.format(z_dim), input_dict = result_dict)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--controller_name', type=str)
    parser.add_argument('--train_types', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--min_z', type=int, default=1)
    parser.add_argument('--max_z', type=int, default=9)
    args = parser.parse_args()

    model_name = args.model_name
    controller_name = args.controller_name
    train_types = args.train_types.split(',')
    weights = args.weights.split(',')
    weights = [float(weight) for weight in weights]
    min_z = args.min_z
    max_z = args.max_z

    BASE_DIR = SCRATCH_DIR + model_name
    DATA_DIR = BASE_DIR + '/dataset/'
    data_dict = load_pkl(DATA_DIR + '/results.pkl')


    controller_val = init_controller(controller_name, data_dict)
    controller_Psi = controller_val.Psi()


    H = data_dict['H']
    u_dim = data_dict['u_dim']
    s_dim = data_dict['s_dim']


    for train_type in train_types:
    
        TEST_DATA_DIR = BASE_DIR + '/pca_results/' + train_type
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

        if train_type == 'weighted':
            for weight in weights:
                TEST_DATA_DIR = BASE_DIR + '/pca_results/' + train_type + '_' + str(weight)
                remove_and_create_dir(TEST_DATA_DIR)
                projection = compute_projection(H, s_dim, controller_Psi, train_type, weight=weight)
                compute_and_save_result(min_z, max_z, projection, data_dict, train_type, TEST_DATA_DIR)
        else:
            projection = compute_projection(H, s_dim, controller_Psi, train_type)
            compute_and_save_result(min_z, max_z, projection, data_dict, train_type, TEST_DATA_DIR)        
                
