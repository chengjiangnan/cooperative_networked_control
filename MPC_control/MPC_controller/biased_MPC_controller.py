import torch
import torch.nn as nn

import numpy as np
from qpth.qp import QPFunction

from recordclass import recordclass

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


BiasedMpcParas = recordclass('BiasedMpcParas', 
                             (
                              'x_dim, '
                              'u_dim, '
                              's_dim, '
                              'A, '
                              'B, '
                              'C, '
                              'Q_h, ' # cost of excess
                              'Q_b, ' # cost of shortage
                              'R, '
                              'H, ' # forecast horizon 
                             )
                            )


class BiasedMpcController(nn.Module):
    def __init__(self, biased_mpc_parameters, device='cpu'):
        """
        Biased MPC Controller adopts biased cost for positive/negative states

        The input matrices are np arrays
        """
        self.x_dim = biased_mpc_parameters.x_dim
        self.u_dim = biased_mpc_parameters.u_dim
        self.s_dim = biased_mpc_parameters.s_dim

        self.A = biased_mpc_parameters.A
        self.B = biased_mpc_parameters.B
        self.C = biased_mpc_parameters.C
        self.Q_h = biased_mpc_parameters.Q_h
        self.Q_b = biased_mpc_parameters.Q_b
        self.R = biased_mpc_parameters.R

        self.H = biased_mpc_parameters.H

        self.device = device


        # the cost function can be simplified as
        # J = 0.5 * x_h_vec.T * Q_h_aug * x_h_vec + \
        #     0.5 * x_h_vec.T * Q_h_aug * x_h_vec + 0.5 * u_vec.T * R_aug * u_vec

        # the dynamics can be expressed as
        # L_aug * x_vec - B_aug * u_vec = C * s_vec + l_aug * x_0

        # the boxed constraints can be expressed as
        # x_h_vec >= x_vec
        # x_b_vec <= x_vec

        self.Q_h_aug = np.zeros((self.H*self.x_dim, self.H*self.x_dim))
        self.Q_b_aug = np.zeros((self.H*self.x_dim, self.H*self.x_dim))
        self.R_aug = np.zeros((self.H*self.u_dim, self.H*self.u_dim))
        self.B_aug = np.zeros((self.H*self.x_dim, self.H*self.u_dim))
        self.C_aug = np.zeros((self.H*self.x_dim, self.H*self.s_dim))
        self.L_aug = np.zeros((self.H*self.x_dim, self.H*self.x_dim))
        for t in range(self.H):
            self.Q_h_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.x_dim:(t+1)*self.x_dim] = self.Q_h
            self.Q_b_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.x_dim:(t+1)*self.x_dim] = self.Q_b
            self.R_aug[t*self.u_dim:(t+1)*self.u_dim, t*self.u_dim:(t+1)*self.u_dim] = self.R
            self.B_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.u_dim:(t+1)*self.u_dim] = self.B
            self.C_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.s_dim:(t+1)*self.s_dim] = self.C
            self.L_aug[t*self.x_dim:(t+1)*self.x_dim, t*self.x_dim:(t+1)*self.x_dim] = np.identity(self.x_dim)
            if t > 0:
                self.L_aug[t*self.x_dim:(t+1)*self.x_dim, (t-1)*self.x_dim:t*self.x_dim] = -self.A

        self.l_aug = np.zeros((self.H*self.x_dim, self.x_dim))
        self.l_aug[:self.x_dim, :] = self.A

        # combine x_h_vec, x_b_vec, u_vec together (for the purpose of using QPFunction)
        # we implicitly use x_vec = x_h_vec + x_b_vec, and
        # x_h_vec >= 0
        # x_l_vec <= 0

        # b_qpth should be computed during runtime by
        # b_qpth = C_aug * s_vec + l_aug * x_0

        self.Q_qpth = np.zeros((self.H*(2*self.x_dim+self.u_dim), self.H*(2*self.x_dim+self.u_dim)))
        self.Q_qpth[:self.H*self.x_dim,:self.H*self.x_dim] = self.Q_h_aug
        self.Q_qpth[self.H*self.x_dim:self.H*2*self.x_dim,self.H*self.x_dim:self.H*2*self.x_dim] = self.Q_b_aug
        self.Q_qpth[self.H*2*self.x_dim:,self.H*2*self.x_dim:] = self.R_aug

        self.p_qpth = np.zeros((self.H*(2*self.x_dim+self.u_dim), 1))

        self.G_qpth = np.zeros((2*self.H*self.x_dim, self.H*(2*self.x_dim+self.u_dim)))
        self.G_qpth[:self.H*self.x_dim, :self.H*self.x_dim] = -np.identity(self.H*self.x_dim)
        self.G_qpth[self.H*self.x_dim:, self.H*self.x_dim:self.H*2*self.x_dim] = np.identity(self.H*self.x_dim)
        
        self.h_qpth = np.zeros((2*self.H*self.x_dim, 1))

        self.A_qpth = np.zeros((self.H*self.x_dim, self.H*(2*self.x_dim+self.u_dim)))
        self.A_qpth[:, :self.H*self.x_dim] = self.L_aug
        self.A_qpth[:, self.H*self.x_dim:self.H*2*self.x_dim] = self.L_aug
        self.A_qpth[:, self.H*2*self.x_dim:] = -self.B_aug

        # create tensors for all the matrices

        self.A = torch.Tensor(self.A).to(device)
        self.B = torch.Tensor(self.B).to(device)
        self.C = torch.Tensor(self.C).to(device)
        self.Q_h = torch.Tensor(self.Q_h).to(device)
        self.Q_b = torch.Tensor(self.Q_b).to(device)
        self.R = torch.Tensor(self.R).to(device)

        self.Q_h_aug = torch.Tensor(self.Q_h_aug).to(device)
        self.Q_b_aug = torch.Tensor(self.Q_b_aug).to(device)
        self.R_aug = torch.Tensor(self.R_aug).to(device)
        self.B_aug = torch.Tensor(self.B_aug).to(device)
        self.C_aug = torch.Tensor(self.C_aug).to(device)
        self.L_aug = torch.Tensor(self.L_aug).to(device)
        self.l_aug = torch.Tensor(self.l_aug).to(device)

        self.Q_qpth = torch.Tensor(self.Q_qpth).to(device)
        self.p_qpth = torch.Tensor(self.p_qpth).to(device).reshape(self.H*(2*self.x_dim+self.u_dim))
        self.G_qpth = torch.Tensor(self.G_qpth).to(device)
        self.h_qpth = torch.Tensor(self.h_qpth).to(device).reshape(2*self.H*self.x_dim)
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

        ## qpth results cannot be trusted, so don't use this piece of code        
        combined_sol = QPFunction(verbose=False)(
            self.Q_qpth, self.p_qpth, self.G_qpth, self.h_qpth, self.A_qpth, b_qpth)

        x_sol = combined_sol[:, :self.H*self.x_dim]+combined_sol[:, self.H*self.x_dim:self.H*2*self.x_dim]
        x_sol = x_sol.reshape(batch_size, self.H, self.x_dim)
        u_sol = combined_sol[:, self.H*2*self.x_dim:].reshape(batch_size , self.H, self.u_dim)

        return x_sol, u_sol
        

    def next_state(self, x0, u0, s0):
        # x0: (batch_size, x_dim, 1)
        # u0: (batch_size, u_dim, 1)
        # s0: (batch_size, s_dim, 1)

        return self.A.matmul(x0) + self.B.matmul(u0) + self.C.matmul(s0)


    def one_step_cost(self, u0, x1):
        # u0: (batch_size, u_dim, 1)
        # x1: (batch_size, x_dim, 1)
        # cost: (batch_size, 1, 1)
        zero_tensor = torch.zeros(x1.shape[0], self.x_dim, 1).to(device)
        x1_h = torch.max(x1, zero_tensor)
        x1_b = torch.min(x1, zero_tensor)

        cost = x1_h.transpose(1, 2).matmul(self.Q_h).matmul(x1_h) + \
               x1_b.transpose(1, 2).matmul(self.Q_b).matmul(x1_b) + \
               u0.transpose(1, 2).matmul(self.R).matmul(u0)

        return cost


if __name__=="__main__":

    x_dim = 2
    u_dim = 2
    s_dim = 2

    A = np.identity(2)
    B = -np.identity(2)
    C = np.identity(2)
    Q_h = 1 * np.identity(2)
    Q_b = 10 * np.identity(2)
    R = np.identity(2)

    H = 3

    mpc_paras = BiasedMpcParas(x_dim, u_dim, s_dim, A, B, C, Q_h, Q_b, R, H)

    model = BiasedMpcController(mpc_paras, device)

    x0_test =  torch.zeros([1, x_dim, 1], dtype=torch.float32)

    forecast_test = -torch.ones([1, H, s_dim], requires_grad = True, dtype=torch.float32)

    x_sol_test, u_sol_test = model.forward(x0_test, forecast_test)

    print("x_sol: {}\nu_sol: {}".format(x_sol_test, u_sol_test))

    u0 = u_sol_test[:, 0:1, :].transpose(1, 2)

    one_step_cost = model.one_step_cost(u0, x0_test)

    print("one step cost: {}".format(one_step_cost))

