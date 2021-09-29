import torch
import torch.nn as nn

import numpy as np

from recordclass import recordclass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


SimpleMpcParas = recordclass('SimpleMpcParas', 
                             (
                              'x_dim, '
                              'u_dim, '
                              's_dim, '  
                              'A, '
                              'B, '
                              'C, '
                              'Q, '
                              'R, '
                              'H, ' # forecast horizon 
                              )
                            )


class SimpleMpcController(nn.Module):
    def __init__(self, simple_mpc_parameters, device='cpu'):
        """
        Simple MPC Controller doesn't contain inequality constraints

        The input matrices are np arrays
        """
        self.x_dim = simple_mpc_parameters.x_dim
        self.u_dim = simple_mpc_parameters.u_dim
        self.s_dim = simple_mpc_parameters.s_dim

        self.A = simple_mpc_parameters.A
        self.B = simple_mpc_parameters.B
        self.C = simple_mpc_parameters.C
        self.Q = simple_mpc_parameters.Q
        self.R = simple_mpc_parameters.R

        self.H = simple_mpc_parameters.H

        self.device = device

        # auxiliary matrices to compute optimal move

        # z = K0.dot(x0) + K1.dot(forecast_vec)
        # u_vec = - inv(Z).dot(z)
        # x_vec = K2.dot(x0) + K3.dot(u_vec) + K4.dot(forecast_vec)
        # cost = x0.T.dot(Q).dot(x0) + x_vec.T.dot(Q_aug).dot(x_vec) + u_vec.T.dot(R_aug).dot(u_vec)

        self.Z = np.zeros((self.H*self.u_dim, self.H*self.u_dim))

        self.K0 = np.zeros((self.H*self.u_dim, self.u_dim))
        self.K1 = np.zeros((self.H*self.u_dim, self.H*self.s_dim))

        self.K2 = np.zeros((self.H*self.x_dim, self.x_dim))
        self.K3 = np.zeros((self.H*self.x_dim, self.H*self.u_dim))
        self.K4 = np.zeros((self.H*self.x_dim, self.H*self.s_dim))

        self.Q_aug = np.zeros((self.H*self.x_dim, self.H*self.x_dim))
        self.R_aug = np.zeros((self.H*self.u_dim, self.H*self.u_dim))


        for t in range(self.H):
            self.Z[t*self.u_dim:(t+1)*self.u_dim, t*self.u_dim:(t+1)*self.u_dim] = self.R
            self.R_aug[t*self.u_dim:(t+1)*self.u_dim, t*self.u_dim:(t+1)*self.u_dim] = self.R

            self.Q_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.x_dim:(t+1)*self.x_dim] = self.Q

        for t in range(1, self.H+1):
            M = np.zeros((self.x_dim, self.H*self.u_dim))
            N = np.zeros((self.x_dim, self.H*self.s_dim))
            for i in range(t):
                power_of_A = np.linalg.matrix_power(self.A, t-1-i)
                M[:, i*self.u_dim:(i+1)*self.u_dim] = power_of_A.dot(self.B)
                N[:, i*self.s_dim:(i+1)*self.s_dim] = power_of_A.dot(self.C)

            power_of_A = np.linalg.matrix_power(self.A, t)
            self.Z = self.Z + M.transpose().dot(self.Q).dot(M)
            self.K0 = self.K0 + M.transpose().dot(self.Q).dot(power_of_A)
            self.K1 = self.K1 + M.transpose().dot(self.Q).dot(N)

            self.K2[(t-1)*self.x_dim:t*self.x_dim, :] = power_of_A
            self.K3[(t-1)*self.x_dim:t*self.x_dim, :] = M
            self.K4[(t-1)*self.x_dim:t*self.x_dim, :] = N

        self.Z_inv = np.linalg.inv(self.Z)

        self.A = torch.Tensor(self.A).to(device)
        self.B = torch.Tensor(self.B).to(device)
        self.C = torch.Tensor(self.C).to(device)
        self.Q = torch.Tensor(self.Q).to(device)
        self.R = torch.Tensor(self.R).to(device)

        self.Z = torch.Tensor(self.Z).to(device)
        self.Z_inv = torch.Tensor(self.Z_inv).to(device)

        self.K0 = torch.Tensor(self.K0).to(device)
        self.K1 = torch.Tensor(self.K1).to(device)

        self.K2 = torch.Tensor(self.K2).to(device)
        self.K3 = torch.Tensor(self.K3).to(device)
        self.K4 = torch.Tensor(self.K4).to(device)

        self.Q_aug = torch.Tensor(self.Q_aug).to(device)
        self.R_aug = torch.Tensor(self.R_aug).to(device)


    def forward(self, x0, forecast):
        # note: batch_size allows us to solve several MPC problems at once
        # each batch can be data from a different problem
        # H = time horizon
        # x0: (batch_size, x_dim, 1)
        # forecast: (batch_size, H, s_dim)
        # u_sol: (batch_size, H, u_dim) [solution for horizon]
        # x_sol: (batch_size, H, x_dim)
        # cost_sol: (batch_size, 1, 1)

        # transform forecast to vector representation: (batch_size, H*s_dim, 1)
        forecast_vec = forecast.reshape(forecast.shape[0], self.H*self.s_dim, 1)

        z = torch.matmul(self.K0, x0) + torch.matmul(self.K1, forecast_vec)

        u_sol_vec = - torch.matmul(self.Z_inv, z)
        x_sol_vec = torch.matmul(self.K2, x0) + \
                    torch.matmul(self.K3, u_sol_vec) + torch.matmul(self.K4, forecast_vec)

        u_sol = u_sol_vec.reshape(u_sol_vec.shape[0], self.H, self.u_dim)
        x_sol = x_sol_vec.reshape(x_sol_vec.shape[0], self.H, self.x_dim)

        return x_sol, u_sol

    def task_loss(self, u_sol_hat, u_sol_star):
        # u_sol_hat/u_sol_star: (batch_size, H, u_dim)

        # Definition I: J_MPC difference for time horizon H
        u_sol_hat_vec = u_sol_hat.reshape(u_sol_hat.shape[0], self.H*self.u_dim, 1)
        u_sol_star_vec = u_sol_star.reshape(u_sol_star.shape[0], self.H*self.u_dim, 1)
        u_sol_vec_diff = u_sol_hat_vec - u_sol_star_vec

        task_loss = torch.mean(u_sol_vec_diff.transpose(1, 2).matmul(self.Z).matmul(u_sol_vec_diff))

        # u_sol_hat_vec = u_sol_hat[:, 0:1, :].reshape(u_sol_hat.shape[0], self.u_dim, 1)
        # u_sol_star_vec = u_sol_star[:, 0:1, :].reshape(u_sol_star.shape[0], self.u_dim, 1)
        # u_sol_vec_diff = u_sol_hat_vec - u_sol_star_vec

        # task_loss = torch.mean(u_sol_vec_diff.transpose(1, 2).matmul(self.Z[:self.u_dim,:self.u_dim]).matmul(u_sol_vec_diff))
        
        return task_loss

    def next_state(self, x0, u0, s0):
        # x0: (batch_size, x_dim, 1)
        # u0: (batch_size, u_dim, 1)
        # s0: (batch_size, s_dim, 1)

        return self.A.matmul(x0) + self.B.matmul(u0) + self.C.matmul(s0)


    def one_step_cost(self, u0, x1):
        # u0: (batch_size, u_dim, 1)
        # x1: (batch_size, x_dim, 1)
        # cost: (batch_size, 1, 1)

        cost = x1.transpose(1, 2).matmul(self.Q).matmul(x1) + u0.transpose(1, 2).matmul(self.R).matmul(u0)

        return cost

    def W(self):
        # W is a (u_dim, s_dim * H) matrix that transforms forecasts to control
        return - torch.matmul(self.Z_inv, self.K1)[0:self.u_dim, :]

    def W_full(self):
        # W is a (u_dim, s_dim * H) matrix that transforms forecasts to control
        return - torch.matmul(self.Z_inv, self.K1)

    def Psi(self):
        # Psi is the codesign matrix that
        K_inv = self.Z_inv
        L = self.K1
        res = torch.matmul(L.transpose(0, 1), K_inv)
        res = torch.matmul(res, L)
        return res


if __name__=="__main__":

    x_dim = 1
    u_dim = 1
    s_dim = 1

    A = np.identity(1)
    B = np.identity(1)
    C = np.identity(1)
    Q = np.identity(1)
    R = np.identity(1)

    H = 5

    mpc_paras = SimpleMpcParas(x_dim, u_dim, s_dim, A, B, C, Q, R, H)

    model = SimpleMpcController(mpc_paras, device)

    weight = - torch.matmul(model.Z_inv, model.K1)[0:1, :]
    print("weight of the forecasts: {}".format(weight))

