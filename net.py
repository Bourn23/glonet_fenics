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

# class Generator:
#     def __init__(self, params):
#         self.sampling_mode = params.generate_samples_mode
#         # self.beta = torch.randn(1, 1, requires_grad = True, dtype = torch.float64)
#         self.E_0 = params.E_0
#         self.nu_0 = params.nu_0
#         self.E_r = None
#         self.nu_r = None
#         self.mu_, self.beta_ = lame(params.E_0, params.nu_0)
#         self.force_ = params.force
#         self.batch_size_ = params.batch_size_start
#         self.mu = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., self.mu_+torch.rand(1)[0]*10).requires_grad_(True)
#         self.beta = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., self.beta_+torch.rand(1)[0]*10).requires_grad_(True)
#         self.force = torch.DoubleTensor([[params.force]] * params.batch_size_start)#, ruquires_grad = True)

        
#         self.mu_sgd = torch.tensor([[self.mu_]], requires_grad=True, dtype=torch.float64)
#         self.beta_sgd = torch.tensor([[self.beta_]], requires_grad=True, dtype=torch.float64)

#     def parameters(self):
#         return [self.mu_sgd, self.beta_sgd]

#     def params_sgd(self):
#         return [self.mu_sgd, self.beta_sgd, self.force]

#     def generate(self):
#         if self.sampling_mode:
#             # self.mu, self.beta = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.mu_+torch.rand(1)[0]*10).requires_grad_(True), torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.beta_+torch.rand(1)[0]*10).requires_grad_(True)
#             # self.force = torch.DoubleTensor([[self.force_]] * self.batch_size_)#, ruquires_grad = True)
#             self.E_r = self.E_0 / 4 * np.random.randn() + self.E_0
#             self.nu_r = self.nu_0 / 4 * np.random.randn() + self.nu_0
#             self.mu, self.beta = lame(self.E_r, self.nu_r)
            
#             self.mu = torch.tensor([[self.mu]], requires_grad=True, dtype=torch.float64)
#             self.beta = torch.tensor([[self.beta]], requires_grad=True, dtype=torch.float64)
#         return [[self.mu, self.beta, self.force],(self.E_r, self.nu_r)]


class Generator:
    def __init__(self, params, generate_sample):
        # if on, every iteration new samples are created
        self.sampling_mode = generate_sample

        # memory
        self.E_0 = params.E_0
        self.nu_0 = params.nu_0
        self.E_r, self.nu_r = lame(params.E_0, params.nu_0) # E_r == mu_, nu_r == nu_0
        self.batch_size_ = params.batch_size_start
        self.force_ = params.force
        self.first_run = True


        # initialization for first time
        # TODO: Why we have this? This serves as the memory for someting like a Distributed RL training instance. 
        self.mu_fixed = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.E_r+torch.rand(1)[0]*10).requires_grad_(True)
        self.beta_fixed = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.nu_r+torch.rand(1)[0]*10).requires_grad_(True)
        

        # add random noise #TODO: this is for obtaining comparison results for the paper... different noise and standard deviation
        self.mu = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.E_r+torch.rand(1)[0]*10).requires_grad_(True)
        self.beta = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.nu_r+torch.rand(1)[0]*10).requires_grad_(True)
        self.force = torch.DoubleTensor([[self.force_]] * self.batch_size_)
        
    def parameters(self):
        return [self.mu, self.beta, self.force]


    def generate(self):
        #TODO: multi data point generation? (* self.batch_size)
        if self.sampling_mode: #or self.first_run:
            # self.mu, self.beta = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.mu_+torch.rand(1)[0]*10).requires_grad_(True), torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.beta_+torch.rand(1)[0]*10).requires_grad_(True)
            # self.force = torch.DoubleTensor([[self.force_]] * self.batch_size_)#, ruquires_grad = True)

            self.E_r = self.E_0 / 4 * np.random.randn() + self.E_0
            self.nu_r = self.nu_0 / 4 * np.random.randn() + self.nu_0
            self.mu, self.beta = lame(self.E_r, self.nu_r)
            
            self.mu = torch.tensor([[self.mu]], requires_grad=True, dtype=torch.float64)
            self.beta = torch.tensor([[self.beta]], requires_grad=True, dtype=torch.float64)
            
            self.first_run = False

            
        # return [[self.mu, self.beta, self.force],(self.E_r, self.nu_r)]
        return {'mu': self.mu, 'beta': self.beta, 'force': self.force, 'E_r': self.E_r, 'nu_r': self.nu_r}


class Model:
    def __init__(self, params):
        # values to store
        print(params.generate_samples_mode)
        self.generator = Generator(params, params.generate_samples_mode)
        self.init_values = self.generator.parameters()
        self.mu = self.init_values[0]
        self.beta = self.init_values[1]
        self.force = self.init_values[2]

        self.history = []

    def train(self, eng):
        # params: eng; physics engine
        pass
        # run the simulation and obtain values
        # samples = self.generator.generate()
        # effs = eng.Eval_Eff_1D_parallel(data)
        # loss = torch.nn.MSELoss()
        # err = loss(effs, eng.target_deflection)

        # # update local values (mu, beta, history)
        # # do we need this ?? self.mu, self.beta, _ = generator.parameters()
        # optimizer.zero_grad()
        # err = torch.log(loss(s_r, s_0))
        # err.backward(retain_graph=True)
        # self.history = np.vstack([self.history, np.array([self.mu, self.beta, self.err])])

    def evaluate(self, eng):
        # generate images
        samples = self.generator.generate()
        

        # efficiencies of generated images
        effs = eng.Eval_Eff_1D_parallel(data)
        loss = torch.nn.MSELoss()
        error = loss(effs, eng.target_deflection)
        # error = loss(effs.cpu().detach(), eng.target_deflection)

        # get most recent mu and beta values
        mu_sgd, beta_sgd, force = generator.params_sgd()


        # plot histogram
        #TODO: replace utils.plot_histogram with wes' plotting function
        # fig_path = params.output_dir +  '/figures/histogram/Iter{}.png'.format(params.iter) 
        # utils.plot_histogram(error, params.iter, fig_path)

        
        # return error.detach(), v[0], v[1], mu_sgd, beta_sgd
        return

    def generate_data(self):
        pass

    def plot(self):
        pass




# class GPR(Model):
#     def __init__(self, data, params):
#         super().__init__(params)
#         from sklearn.gaussian_process import GaussianProcessRegressor
#         from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF


#         self.ls = np.std(data, axis = 0)[:2]
#         self.kernel = DotProduct() + WhiteKernel() + RBF(self.ls)
#         self.gpr = GaussianProcessRegressor(kernel = self.kernel).fit(data[:, :2], np.log(data[:,2]))

#         self.X, self.Y = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 11),
#                         np.linspace(data[:, 1].min(), data[:, 1].max(), 11))

#         XY = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
#         self.Z, self.U = self.gpr.predict(XY, return_std = True)


#         from scipy.optimize import minimize
#         self. A = self.gp_ucb(XY)

#         # find the maximal value in the acquisition function
#         self.best = np.argmax(A)
#         self.x0 = XY[self.best]

#         # find the optimal value from this regressor
#         self.res = minimize(self.gp_ucb, self.x0)

#         self.next = self.res.x

#     def train(self, data):
#         self.ls = np.std(data, axis = 0)[:2]
#         self.kernel = DotProduct() + WhiteKernel() + RBF(self.ls)
#         self.gpr = GaussianProcessRegressor(kernel = self.kernel).fit(data[:, :2], np.log(data[:,2]))

#         self.X, self.Y = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), 11),
#                         np.linspace(data[:, 1].min(), data[:, 1].max(), 11))

#         XY = np.hstack([X.reshape(-1, 1), Y.reshape(-1, 1)])
#         self.Z, self.U = self.gpr.predict(XY, return_std = True)



#         self. A = self.gp_ucb(XY)

#         # find the maximal value in the acquisition function
#         self.best = np.argmax(A)
#         self.x0 = XY[self.best]

#         # find the optimal value from this regressor
#         self.res = minimize(self.gp_ucb, self.x0)

#         self.next = self.res.x


#     def plot(self, fig_path):
#         fig, ax = plt.subplots(1, 3, figsize = (9, 3))
    
#         ax[0].set_title('Predicted loss')
#         ax[0].contourf(self.X, self.Y, self.Z.reshape(self.X.shape))
#         ax[0].plot(self.E_0, self.nu_0, 'ws')  # white = true value
#         ax[0].plot(*self.next, 'rs')  # red = predicted value

#         ax[1].set_title('Uncertainty')
#         ax[1].contourf(self.X, self.Y, self.U.reshape(self.X.shape))

#         ax[2].set_title('Acquisition function')
#         ax[2].contourf(self.X, self.Y, self.A.reshape(self.X.shape))

#         plt.savefig(fig_path, dpi=300)
#         plt.close()

#     def gp_ucb(self, x):
#         "acquisition function, maximize upper confidence bound (GP-UCB) "
#         if len(x.shape) < 2:
#             x = [x]
#         self.Z, self.U = self.gpr.predict(x, return_std=True)
#         return -self.Z + 1e-6*self.U

class SGD(Model):
    def __init__(self, params):
        super().__init__(params)
        self.optimizer = torch.optim.Adam(self.generator.parameters()[:-1], lr=params.lr, betas=(params.beta1, params.beta2))

    def train(self, eng):
        data = self.generator.generate()
        effs = eng.Eval_Eff_1D_parallel(data)
        loss = torch.nn.MSELoss()

        # update local values (mu, beta, history)
        # do we need this ?? self.mu, self.beta, _ = generator.parameters()
        self.optimizer.zero_grad()
        err = torch.log(loss(effs, eng.target_deflection))
        err.backward(retain_graph=True)
        self.history = np.vstack([self.history, np.array([self.mu.detach(), self.beta.detach(), self.err.detach()])])  



class SGD_Updater:
    def __init__(self, data, params, generator):
        # Define the optimizer
        sellf.optimizer = torch.optim.Adam(self.parameters(), lr=params.lr, betas=(params.beta1, params.beta2))
        
        # Define the scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=params.step_size, gamma = params.gamma)

        self.data = data

        self.mu = self.generator.mu_fixed # do we need a deep copy here?
        self.beta = self.generator.beta_fixed
        self.force = self.generator.force_fixed

    def parameters(self):
        return [self.mu, self.beta]
        

    def plot(self, fig_path):
        fig, ax = plt.subplots()

        ax.contourf(self.X, self.Y, self.Z.reshape(self.X.shape))
        ax.plot(self.data[:, 0], self.data[:, 1], 'rx')  # values obtained by torch
        ax.plot(self.E_0, self.nu_0, 'ws')  # white = true value

        plt.savefig(fig_path, dpi = 300)
        plt.close()

    def train(self):
        pass