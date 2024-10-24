import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import copy
from scipy.integrate import odeint

dtype = torch.float
import torch.nn as nn
import math

# 检查MPS是否可用
if torch.backends.mps.is_available():
    print("MPS is available")

    # 设置设备为MPS
    device = torch.device("mps")

    # 输出当前设备（MPS设备）
    print("Current device:", device)

    # 因为MPS是单一设备，因此无法像CUDA那样获取多个设备的信息
    print("Device count: 1")

    # 获取设备的名称（通常是Apple设备的GPU）
    print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Apple MPS")
else:
    print("MPS is not available, using CPU")
    device = torch.device("cpu")

# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


# 利用pytorch自动计算f对x的导数，x为输入张量，f为输出张量，返回df/dx
# Define some more general functions
def dfx(x, f):
    # Calculate the derivative with auto-differentiation
    gopts = torch.ones(x.shape, dtype=dtype,device=device)
    return grad([f], [x], grad_outputs=gopts, create_graph=True)[0]

# 神经网络的结构定义
class qNN1(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(qNN1, self).__init__()

        # Define the Activation
        #  self.actF = torch.nn.Sigmoid()
        self.actF = mySin()

        # define layers
        # self.Lin_1   = torch.nn.Linear(1, D_hid)
        # self.E_out = torch.nn.Linear(D_hid, 1)
        # self.Lin_2 = torch.nn.Linear(D_hid, D_hid)
        # self.Ein = torch.nn.Linear(1,1)
        # self.Lin_out = torch.nn.Linear(D_hid+1, 1)
        self.sym = True
        self.Ein = torch.nn.Linear(1, 1)
        self.Lin_1 = torch.nn.Linear(2, D_hid)
        self.Lin_2 = torch.nn.Linear(D_hid + 1, D_hid)
        self.out = torch.nn.Linear(D_hid + 1, 1)

    # 返回的是列表[N(x,lambda),lambda]
    def forward(self, t):
        In1 = self.Ein(torch.ones_like(t))

        L1 = self.Lin_1(torch.cat((t, In1), 1))
        L1p = self.Lin_1(torch.cat((-1 * t, In1), 1))

        h1 = self.actF(L1)
        h1p = self.actF(L1p)

        L2 = self.Lin_2(torch.cat((h1, In1), 1))
        L2p = self.Lin_2(torch.cat((h1p, In1), 1))

        h2 = self.actF(L2)
        h2p = self.actF(L2p)

        # out = self.out(torch.cat((h2+h2p,In1),1))
        # out = self.out(torch.cat((h2,In1),1))

        # 根据系统的宇称的奇偶性判断波函数的对称性
        if self.sym:
            out = self.out(torch.cat((h2 + h2p, In1), 1))
        else:
            out = self.out(torch.cat((h2 - h2p, In1), 1))

        return out, In1


class PotentialBase():
    # x in [t0,tf]，neurons为隐藏层的神经元的数量，epochs为学习次数，n_train为在采样区间内采样的数量，lr为学习率
    def __init__(self, t0, tf, x1, neurons, epochs, n_train, lr, minibatch_number=1):
        self._t0 = t0
        self._tf = tf
        self._x1 = x1
        self._neurons = neurons
        self._epochs = epochs
        self._n_train = n_train
        self._lr = lr
        self._minibatch_number = minibatch_number
        # 用来记录所有的loss
        self._loss_history = ()

    def _potential(self, Xs):
        raise NotImplementedError

    def _weights_init(self, m):
        if isinstance(m, nn.Linear) and m.weight.shape[0] != 1:
            torch.nn.init.xavier_uniform(m.weight.data)

    # Train the NN
    # x in [t0,tf]，neurons为隐藏层的神经元的数量，epochs为学习次数，n_train为在采样区间内采样的数量，lr为学习率
    def train(self):
        raise NotImplementedError

    # visualization the wavefunction represented by nn, t is the x-parameters, x1 is f_b in the paper
    def parametricSolutions(self, t, nn):
        N1, N2 = nn(t)
        dt = t - self._t0
        f = (1 - torch.exp(-dt)) * (1 - torch.exp(dt - 12))
        psi_hat = self._x1 + f * N1
        return psi_hat

    def _perturbPoints(self, grid, sig):
        # stochastic perturbation of the evaluation points
        # force t[0]=t0  & force points to be in the t-interval
        delta_t = grid[1] - grid[0]
        noise = delta_t * torch.randn_like(grid) * sig
        t = grid + noise
        t.data[2] = torch.ones(1, 1) * (-1)
        t.data[t < self._t0] = self._t0 - t.data[t < self._t0]
        t.data[t > self._tf] = 2 * self._tf - t.data[t > self._tf]
        t.data[0] = torch.ones(1, 1) * self._t0

        t.data[-1] = torch.ones(1, 1) * self._tf
        t.requires_grad = False
        return t

    # t is the set of sample points, f is the residual of the Eigen equation, L is the L_DE, H_psi is the Hamiltonian
    def _hamEqs_Loss(self, t, psi, E, V):
        psi_dx = dfx(t, psi)
        psi_ddx = dfx(t, psi_dx)
        f = psi_ddx / 2 + (E - V) * psi
        L = (f.pow(2)).mean();
        H_psi = -1 * psi_ddx / 2 + (V) * psi
        return L, f, H_psi



