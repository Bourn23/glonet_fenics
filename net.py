import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 
import matplotlib.pyplot as plt
from utils import lame, youngs_poisson

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Generator:
    def __init__(self, params):
        self.sampling_mode = params.generate_samples_mode
        # self.beta = torch.randn(1, 1, requires_grad = True, dtype = torch.float64)
        self.E_0 = params.E_0
        self.nu_0 = params.nu_0
        self.E_r = None
        self.nu_r = None
        self.mu_, self.beta_ = lame(params.E_0, params.nu_0)
        self.force_ = params.force
        self.batch_size_ = params.batch_size_start
        self.mu = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., self.mu_+torch.rand(1)[0]*10).requires_grad_(True)
        self.beta = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., self.beta_+torch.rand(1)[0]*10).requires_grad_(True)
        self.force = torch.DoubleTensor([[params.force]] * params.batch_size_start)#, ruquires_grad = True)
    
    def parameters(self):
        return [self.mu, self.beta]

    def generate(self):
        if self.sampling_mode:
            # self.mu, self.beta = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.mu_+torch.rand(1)[0]*10).requires_grad_(True), torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.beta_+torch.rand(1)[0]*10).requires_grad_(True)
            # self.force = torch.DoubleTensor([[self.force_]] * self.batch_size_)#, ruquires_grad = True)
            self.E_r = self.E_0 / 4 * np.random.randn() + self.E_0
            self.nu_r = self.nu_0 / 4 * np.random.randn() + self.nu_0
            self.mu, self.beta = lame(self.E_r, self.nu_r)
            
            self.mu = torch.tensor([[self.mu]], requires_grad=True, dtype=torch.float64)
            self.beta = torch.tensor([[self.beta]], requires_grad=True, dtype=torch.float64)
        return [[self.mu, self.beta, self.force],(E_r, nu_r)]

def gp_ucb(x):
    global gpr
    if len(x.shape) < 2:
        x = [x]
    Z, U = gpr.predict(x, return_std=True)
    return -Z + 1e-6*U



def GPR(data, params, fig_path):
    global gpr
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

    ls = np.std(data, axis = 0)[:2]
    kernel = DotProduct() + WhiteKernel() + RBF(ls)
    gpr = GaussianProcessRegressor(kernel = kernel).fit(data[:, :2], np.log(data[:,2]))

    X, Y = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 11),
                       np.linspace(data[:, 1].min(), data[:, 1].max(), 11))

    XY = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
    Z, U = gpr.predict(XY, return_std = True)

    from scipy.optimize import minimize
# from google.colab import output

# acquisition function, maximize upper confidence bound (GP-UCB) 


    A = gp_ucb(XY)

    # find the maximal value in the acquisition function
    best = np.argmax(A)
    x0 = XY[best]

    # find the optimal value from this regressor
    res = minimize(gp_ucb, x0)

    next = res.x

    # plotting shit
    fig, ax = plt.subplots(1, 3, figsize = (9, 3))
    
    ax[0].set_title('Predicted loss')
    ax[0].contourf(X, Y, Z.reshape(X.shape))
    # ax[0].plot(E_0, nu_0, 'ws')  # white = true value
    ax[0].plot(params.E_0, params.nu_0, 'ws')  # white = true value
    ax[0].plot(*next, 'rs')  # red = predicted value

    ax[1].set_title('Uncertainty')
    ax[1].contourf(X, Y, U.reshape(X.shape))

    ax[2].set_title('Acquisition function')
    ax[2].contourf(X, Y, A.reshape(X.shape))

    plt.savefig(fig_path, dpi=300)
    plt.close()

# next version is a class
