import torch
import torch.nn as nn

import numpy as np
from qpth.qp import QPFunction

from recordclass import recordclass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BoxMpcParas = recordclass('BoxMpcParas', 
                          (
                           'x_dim, '
                           'u_dim, '
                           's_dim, '
 
                           'u_ub, '
                           'u_lb, '

                           'A, '
                           'B, '
                           'C, '
                           'Q, '
                           'R, '
                           'H, ' # forecast horizon 
                           )
                         )


class BoxMpcController(nn.Module):
    def __init__(self, box_mpc_parameters, device='cpu'):
        """
        Box MPC Controller contains boxed inequality constraints

        The input matrices are np arrays
        """
        self.x_dim = box_mpc_parameters.x_dim
        self.u_dim = box_mpc_parameters.u_dim
        self.s_dim = box_mpc_parameters.s_dim

        self.u_ub = box_mpc_parameters.u_ub
        self.u_lb = box_mpc_parameters.u_lb

        self.A = box_mpc_parameters.A
        self.B = box_mpc_parameters.B
        self.C = box_mpc_parameters.C
        self.Q = box_mpc_parameters.Q
        self.R = box_mpc_parameters.R

        self.H = box_mpc_parameters.H

        self.device = device

        self.u_ub_aug = self.u_ub * np.ones((self.H*self.u_dim, 1))
        self.u_lb_aug = self.u_lb * np.ones((self.H*self.u_dim, 1))

        # the cost function can be simplified as
        # J = 0.5 * x_vec.T * Q_aug * x_vec + 0.5 * u_vec.T * R_aug * u_vec

        # the dynamics can be expressed as
        # L_aug * x_vec - l_aug * x_0 - B_aug * u_vec = C * s_vec

        # the boxed constraints can be expressed as
        # u_lb_aug <= u_vec <= u_ub_aug
        # x doesn't have constraints

        self.Q_aug = np.zeros((self.H*self.x_dim, self.H*self.x_dim))
        self.R_aug = np.zeros((self.H*self.u_dim, self.H*self.u_dim))
        self.B_aug = np.zeros((self.H*self.x_dim, self.H*self.u_dim))
        self.C_aug = np.zeros((self.H*self.x_dim, self.H*self.s_dim))
        self.L_aug = np.zeros((self.H*self.x_dim, self.H*self.x_dim))
        for t in range(self.H):
            self.Q_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.x_dim:(t+1)*self.x_dim] = self.Q
            self.R_aug[t*self.u_dim:(t+1)*self.u_dim, t*self.u_dim:(t+1)*self.u_dim] = self.R
            self.B_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.u_dim:(t+1)*self.u_dim] = self.B
            self.C_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.s_dim:(t+1)*self.s_dim] = self.C
            self.L_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.x_dim:(t+1)*self.x_dim] = np.identity(self.x_dim)
            if t > 0:
                self.L_aug[t*self.x_dim:(t+1)*self.x_dim, (t-1)*self.x_dim:t*self.x_dim] = -self.A

        self.l_aug = np.zeros((self.H*self.x_dim, self.x_dim))
        self.l_aug[:self.x_dim, :] = self.A

        # combine x_vec and u_vec together (for the purpose of using QPFunction)
        # b_qpth should be computed during runtime by
        # b_qpth = C_aug * s_vec + l_aug * x_0

        self.Q_qpth = np.zeros((self.H*(self.x_dim+self.u_dim), self.H*(self.x_dim+self.u_dim)))
        self.Q_qpth[:self.H*self.x_dim,:self.H*self.x_dim] = self.Q_aug
        self.Q_qpth[self.H*self.x_dim:, self.H*self.x_dim:] = self.R_aug

        self.p_qpth = np.zeros((self.H*(self.x_dim+self.u_dim), 1))

        self.G_qpth = np.zeros((2*self.H*self.u_dim, self.H*(self.x_dim+self.u_dim)))
        self.G_qpth[:self.H*self.u_dim, self.H*self.x_dim:] = np.identity(self.H*self.u_dim)
        self.G_qpth[self.H*self.u_dim:, self.H*self.x_dim:] = -np.identity(self.H*self.u_dim)
        
        self.h_qpth = np.zeros((2*self.H*self.u_dim, 1))
        self.h_qpth[:self.H*self.u_dim, :] = self.u_ub_aug
        self.h_qpth[self.H*self.u_dim:, :] = -self.u_lb_aug

        self.A_qpth = np.zeros((self.H*self.x_dim, self.H*(self.x_dim+self.u_dim)))
        self.A_qpth[:, :self.H*self.x_dim] = self.L_aug
        self.A_qpth[:, self.H*self.x_dim:] = -self.B_aug

        # create tensors for all the matrices

        self.A = torch.Tensor(self.A).to(device)
        self.B = torch.Tensor(self.B).to(device)
        self.C = torch.Tensor(self.C).to(device)
        self.Q = torch.Tensor(self.Q).to(device)
        self.R = torch.Tensor(self.R).to(device)

        self.u_ub_aug = torch.Tensor(self.u_ub_aug).to(device)
        self.u_lb_aug = torch.Tensor(self.u_lb_aug).to(device)

        self.Q_aug = torch.Tensor(self.Q_aug).to(device)
        self.R_aug = torch.Tensor(self.R_aug).to(device)
        self.B_aug = torch.Tensor(self.B_aug).to(device)
        self.C_aug = torch.Tensor(self.C_aug).to(device)
        self.L_aug = torch.Tensor(self.L_aug).to(device)
        self.l_aug = torch.Tensor(self.l_aug).to(device)

        self.Q_qpth = torch.Tensor(self.Q_qpth).to(device)
        self.p_qpth = torch.Tensor(self.p_qpth).to(device).reshape(self.H*(self.x_dim+self.u_dim))
        self.G_qpth = torch.Tensor(self.G_qpth).to(device)
        self.h_qpth = torch.Tensor(self.h_qpth).to(device).reshape(2*self.H*self.u_dim)
        self.A_qpth = torch.Tensor(self.A_qpth).to(device)


    def forward(self, x0, forecast):
        # note: batch_size allows us to solve several MPC problems at once
        # each batch can be data from a different problem
        # H = time horizon
        # x0: (batch_size, x_dim, 1)
        # forecast: (batch_size, H, s_dim)
        # u_sol: (batch_size, H, u_dim) [solution for horizon]
        # x_sol: (batch_size, H, x_dim)

        # transform forecast to vector representation: (batch_size, H*s_dim, 1)
        batch_size = forecast.shape[0]
        forecast_vec = forecast.reshape(batch_size, self.H*self.s_dim, 1)

        b_qpth = self.C_aug.matmul(forecast_vec) + self.l_aug.matmul(x0)
        b_qpth = b_qpth.reshape(batch_size, self.H*self.x_dim)

        combined_sol = QPFunction(verbose=False)(
            self.Q_qpth, self.p_qpth, self.G_qpth, self.h_qpth, self.A_qpth, b_qpth)

        x_sol = combined_sol[:, :self.H*self.x_dim].reshape(batch_size, self.H, self.x_dim)
        u_sol = combined_sol[:, self.H*self.x_dim:].reshape(batch_size , self.H, self.u_dim)

        return x_sol, u_sol

    def task_loss(self, u_sol_hat, x_sol_hat, u_sol_star, x_sol_star):
        return None

    def next_state(self, x0, u0, s0):
        # x0: (batch_size, x_dim, 1)
        # u0: (batch_size, u_dim, 1)
        # s0: (batch_size, s_dim, 1)

        return self.A.matmul(x0) + self.B.matmul(u0) + self.C.matmul(s0)


    def one_step_cost(self, u0, x1):
        # u0: (batch_size, u_dim, 1)
        # x1: (batch_size, x_dim, 1)
        # cost: (batch_size, 1, 1)
        cost = x1.transpose(1, 2).matmul(self.Q).matmul(x1) + \
               u0.transpose(1, 2).matmul(self.R).matmul(u0)

        return cost


if __name__=="__main__":

    x_dim = 2
    u_dim = 2
    s_dim = 2

    u_ub = 0.9
    u_lb = 0

    A = np.identity(2)
    B = -np.identity(2)
    C = np.identity(2)
    Q = np.identity(2)
    R = np.identity(2)

    H = 3

    mpc_paras = BoxMpcParas(x_dim, u_dim, s_dim, u_ub, u_lb, A, B, C, Q, R, H)

    model = BoxMpcController(mpc_paras, device)

    x0_test =  torch.zeros([1, x_dim, 1], dtype=torch.float32)

    forecast_test = torch.ones([1, H, s_dim], requires_grad = True, dtype=torch.float32)

    x_sol_test, u_sol_test = model.forward(x0_test, forecast_test)

    print("x_sol: {}\nu_sol: {}".format(x_sol_test, u_sol_test))

