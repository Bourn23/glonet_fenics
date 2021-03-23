import sys
import time

sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 
import matplotlib.pyplot as plt
from utils import lame, youngs_poisson, make_gif_from_folder

# GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV


# DEAP (GA/PSO)
from deap import base, creator, algorithms, benchmarks
from deap import tools
import random
import operator
import math


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Generator:
    def __init__(self, params, generate_sample):
        # if on, every iteration new samples are created
        self.sampling_mode = generate_sample

        # memory
        self.E_0 = params.E_0
        self.nu_0 = params.nu_0
        self.E_r, self.nu_r = lame(params.E_0, params.nu_0) # E_r == mu_, nu_r == nu_0
        self.batch_size_ = params.batch_size_start
        self.force = self.force_ = params.force

        # perturb initial values
        self.E_r = self.E_0 / 4 * np.random.randn() + self.E_0
        self.nu_r = self.nu_0 / 4 * np.random.randn() + self.nu_0
        
        self.force = torch.DoubleTensor([[self.force_]])
        self.mu, self.beta = lame(self.E_r, self.nu_r)
        self.mu = torch.tensor([[self.mu]], requires_grad=True, dtype=torch.float64)
        self.beta = torch.tensor([[self.beta]], requires_grad=True, dtype=torch.float64)
        
    def parameters(self):
        return [self.mu, self.beta, self.force]


    def generate(self, sampling = False):
        #TODO: multi data point generation? (* self.batch_size)
        if self.sampling_mode or sampling: #or self.first_run:
            self.E_r = self.E_0 / 4 * np.random.randn() + self.E_0
            self.nu_r = self.nu_0 / 4 * np.random.randn() + self.nu_0
            self.mu, self.beta = lame(self.E_r, self.nu_r)
            
            self.mu = torch.tensor([[self.mu]], requires_grad=True, dtype=torch.float64)
            self.beta = torch.tensor([[self.beta]], requires_grad=True, dtype=torch.float64)
            
            self.first_run = False
            
        return {'mu': self.mu, 'beta': self.beta, 'force': self.force, 'E_r': self.E_r, 'nu_r': self.nu_r}


class Model:
    def __init__(self, params):
        """
        :param params: a list of model related variables
        
        :Example:
        >>> class SGD(Model):

        :Keywords:
        model, base model
        """
        # values to store
        self.generator = Generator(params, params.generate_samples_mode)
        
        self.history = np.zeros([0,3])
        self.loss_history = np.zeros([0,2])
        self.data    = np.zeros([0,3])

        self.training_time = 0
        self.infer_time = 0

    def train(self, eng):
        """
        repeated every params.numIter
        :param eng: API to the compiled problem in physics engine
        
        :Keywords:
        training, model training
        """
        # params: eng; physics engine
        pass

    def evaluate(self, eng):
        """
        the evaluation function is called every params.eval_iter
        :param eng: API to the compiled problem in physics engine
        
        :Keywords:
        evaluation, 
        """
        pass

    def generate_data(self):

        pass

    def plot(self, fig_path, global_memory, axis = None):
        """
        the evaluation function is called every params.eval_iter
        :param fig_path: storage directory
        :param global_memory: access to other models' shared states and information
        
        :Keywords:
        evaluation, 
        """
        pass
    
    def summary(self):
        """
        after finishing the entire optimization, is called to provide a statistics of the solver
        
        
        :Keywords:
        summarization, model statistics
        """
        pass

class GPR(Model):
    def __init__(self, params, eng, global_memory, model_params = None):
        super().__init__(params)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
        from scipy.optimize import minimize
        
        # loading from memory:
        # try: self.data = global_memory.gpr_data
        # except: pass

        self.loss = nn.MSELoss()
        # init_data(params.gpr_init)
        self.init_data(eng, 50) #TODO: chnage it to 1


    def init_data(self, eng, i  = 200):
        # start_time = time.time()

        for i in range(i):
            parameters = self.generator.generate(sampling = True)

            pred_deflection = eng.Eval_Eff_1D_parallel(parameters)
            err = self.loss(pred_deflection, eng.target_deflection).detach().numpy()
            
            # build internal memory
            self.data = np.vstack([self.data, np.array([parameters['E_r'], parameters['nu_r'], err])])

        end_time = time.time()
        # self.training_time += end_time - start_time
    # @staticmethod
    

    def train(self, eng, t, global_memory):
        # self.init_data(eng, 1)
        global_memory.gpr_data = self.data
        
        start_time = time.time()
        
        #TODO: moved the training process to eval
        ls = np.std(self.data, axis=0)[:2]
        
        kernel = DotProduct() + WhiteKernel() + RBF(ls)
        self.gpr = GaussianProcessRegressor(kernel=kernel).fit(self.data[:, :2], np.log(self.data[:, 2]))

        self.X, self.Y = np.meshgrid(np.linspace(self.data[:, 0].min(), self.data[:, 0].max(), 11),
                        np.linspace(self.data[:, 1].min(), self.data[:, 1].max(), 11))

        self.XY = np.hstack([self.X.reshape(-1, 1), self.Y.reshape(-1, 1)])
        self.Z, self.U = self.gpr.predict(self.XY, return_std=True)

 
        # acquisition function, maximize upper confidence bound (GP-UCB) 
        def gp_ucb(x):
            if len(x.shape) < 2:
                x = [x]
            Z, U = self.gpr.predict(x, return_std=True)
            return -Z + 1e-6*U

        def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
            '''
            Computes the EI at points X based on existing samples X_sample
            and Y_sample using a Gaussian process surrogate model.
            
            Args:
                X: Points at which EI shall be computed (m x d).
                X_sample: Sample locations (n x d).
                Y_sample: Sample values (n x 1).
                gpr: A GaussianProcessRegressor fitted to samples.
                xi: Exploitation-exploration trade-off parameter.
            
            Returns:
                Expected improvements at points X.
            '''
            mu, sigma = gpr.predict(X, return_std=True)
            mu_sample = gpr.predict(X_sample)

            sigma = sigma.reshape(-1, 1)
            
            # Needed for noise-based model,
            # otherwise use np.max(Y_sample).
            # See also section 2.4 in [1]
            mu_sample_opt = np.max(mu_sample)

            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0

            return ei

        
        self.A = gp_ucb(self.XY)

        # find the maximal value in the acquisition function
        best = np.argmax(self.A)
        x0 = self.XY[best]

        # find the optimal value from this regressor
        self.res = minimize(gp_ucb, x0)
        self.next = self.res.x

        # calc error for proposed points
        mu = torch.tensor([[self.next[0]]], requires_grad=True, dtype=torch.float64)
        beta = torch.tensor([[self.next[1]]], requires_grad=True, dtype=torch.float64)
        pred_deflection = eng.Eval_Eff_1D_parallel({'mu': mu, 'beta': beta})
        err = self.loss(pred_deflection, eng.target_deflection).detach().numpy()
        
        # adding next point to data
        self.data = np.vstack([self.data, np.array([self.next[0], self.next[1], err])])


        end_time = time.time()
        self.training_time += end_time - start_time

        # adding to global state
        global_memory.gpr_X = self.X
        global_memory.gpr_Y = self.Y
        global_memory.gpr_Z = self.Z
        global_memory.gpr_XY = self.XY

    #TODO: TOTHINK, why  don't we do all these following commands in the evaluate?


    def plot(self, fig_path, global_memory, axis = None):
        if axis:
            axis.set_title('Predicted loss \n GPR')
            axis.contourf(self.X, self.Y, self.Z.reshape(self.X.shape))
            l, = axis.plot(self.generator.E_0, self.generator.nu_0, 'bs')  # white = true value
            l,  = axis.plot(*self.next, 'rs')  # red = predicted value
            return l
            

        fig, ax = plt.subplots(1, 3, figsize=(6, 3))

        ax[0].set_title('Predicted loss')
        ax[0].contourf(self.X, self.Y, self.Z.reshape(self.X.shape))
        ax[0].plot(self.generator.E_0, self.generator.nu_0, 'bs')  # white = true value
        ax[0].plot(*self.next, 'rs')  # red = predicted value

        ax[1].set_title('Uncertainty')
        ax[1].contourf(self.X, self.Y, self.U.reshape(self.X.shape))

        ax[2].set_title('Acquisition function')
        ax[2].contourf(self.X, self.Y, self.A.reshape(self.X.shape))


        plt.savefig(fig_path, dpi = 300)
        plt.close()

    def summary(self, global_memory): # TODO: convert this table to a function..
        Z = -self.gpr.predict(self.XY, return_std=False)
        Z = Z.reshape(self.X.shape)

        mu = np.max(Z[:, 0])
        beta = np.max(Z[:, 1])
        # plot
        E_f, nu_f = youngs_poisson(mu, beta)
        relative_E_error = (E_f* 10**5-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100
        print("\n--------------GPR---------------")
        print('elapsed time:    {:.2f} (s)'.format(self.training_time))
        print('ground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error,
                                                        relative_nu_error))
        
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])

    def evaluate(self, global_memory):
        pass
        # global_memory.gpr_data = self.data
        
        # start_time = time.time()
        
        # #TODO: moved the training process to eval
        # ls = np.std(self.data, axis=0)[:2]
        
        # kernel = DotProduct() + WhiteKernel() + RBF(ls)
        # self.gpr = GaussianProcessRegressor(kernel=kernel).fit(self.data[:, :2], np.log(self.data[:, 2]))

        # self.X, self.Y = np.meshgrid(np.linspace(self.data[:, 0].min(), self.data[:, 0].max(), 11),
        #                 np.linspace(self.data[:, 1].min(), self.data[:, 1].max(), 11))

        # self.XY = np.hstack([self.X.reshape(-1, 1), self.Y.reshape(-1, 1)])
        # self.Z, self.U = self.gpr.predict(self.XY, return_std=True)

 
        # # acquisition function, maximize upper confidence bound (GP-UCB) 
        # def gp_ucb(x):
        #     if len(x.shape) < 2:
        #         x = [x]
        #     Z, U = self.gpr.predict(x, return_std=True)
        #     return -Z + 1e-6*U

        
        # self.A = gp_ucb(self.XY)

        # # find the maximal value in the acquisition function
        # best = np.argmax(self.A)
        # x0 = self.XY[best]

        # # find the optimal value from this regressor
        # self.res = minimize(gp_ucb, x0)
        # self.next = self.res.x
        # print(f'let\'s go to {self.next} next')

        # end_time = time.time()
        # self.training_time += end_time - start_time

        # # adding to global state
        # global_memory.gpr_X = self.X
        # global_memory.gpr_Y = self.Y
        # global_memory.gpr_Z = self.Z
        # global_memory.gpr_XY = self.XY


class GPRL(Model):
    def __init__(self, params, eng, global_memory, model_params = None):
        super().__init__(params)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
        from scipy.optimize import minimize

        self.mode = 'EI'

        self.loss = nn.MSELoss()
        # init_data(params.gpr_init)
        self.init_data(eng, 50) #TODO: chnage it to 1


    def init_data(self, eng, i  = 200):
        # start_time = time.time()

        for i in range(i):
            parameters = self.generator.generate(sampling = True)

            pred_deflection = eng.Eval_Eff_1D_parallel(parameters)
            err = self.loss(pred_deflection, eng.target_deflection).detach().numpy()
            
            # build internal memory
            self.data = np.vstack([self.data, np.array([parameters['E_r'], parameters['nu_r'], err])])

        end_time = time.time()
        # self.training_time += end_time - start_time
    # @staticmethod
    

    def train(self, eng, t, global_memory):
        # self.init_data(eng, 1)
        global_memory.gpr_data = self.data
        
        start_time = time.time()
        
        #TODO: moved the training process to eval
        ls = np.std(self.data, axis=0)[:2]
        
        kernel = DotProduct() + WhiteKernel() + RBF(ls)
        self.gpr = GaussianProcessRegressor(kernel=kernel).fit(self.data[:, :2], np.log(self.data[:, 2]))

        self.X, self.Y = np.meshgrid(np.linspace(self.data[:, 0].min(), self.data[:, 0].max(), 11),
                        np.linspace(self.data[:, 1].min(), self.data[:, 1].max(), 11))

        self.XY = np.hstack([self.X.reshape(-1, 1), self.Y.reshape(-1, 1)])
        self.Z, self.U = self.gpr.predict(self.XY, return_std=True)

 
        # acquisition function, maximize upper confidence bound (GP-UCB) 
        def gp_ucb(x):
            if len(x.shape) < 2:
                x = [x]
            Z, U = self.gpr.predict(x, return_std=True)
            return -Z + 1e-6*U

        def expected_improvement(X, gpr, xi=0.01):
            '''
            Computes the EI at points X based on existing samples X_sample
            and Y_sample using a Gaussian process surrogate model.
            
            Args:
                X: Points at which EI shall be computed (m x d).
                X_sample: Sample locations (n x d).
                Y_sample: Sample values (n x 1).
                gpr: A GaussianProcessRegressor fitted to samples.
                xi: Exploitation-exploration trade-off parameter.
            
            Returns:
                Expected improvements at points X.
            '''

            if len(X.shape) < 2:
                X = [X]
            mu, sigma = gpr.predict([X[-1]], return_std=True)
            mu_sample = gpr.predict(X)

            sigma = sigma.reshape(-1, 1)
            
            # Needed for noise-based model,
            # otherwise use np.max(Y_sample).
            # See also section 2.4 in [1]
            mu_sample_opt = np.max(mu_sample)

            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0

            return ei

        def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
            '''
            Proposes the next sampling point by optimizing the acquisition function.
            
            Args:
                acquisition: Acquisition function.
                X_sample: Sample locations (n x d).
                Y_sample: Sample values (n x 1).
                gpr: A GaussianProcessRegressor fitted to samples.

            Returns:
                Location of the acquisition function maximum.
            '''
            dim = X_sample.shape[1]
            min_val = 1
            min_x = None
            
            def min_obj(X):
                # Minimization objective is the negative acquisition function
                return -acquisition([X[-1]], X_sample, Y_sample, gpr)
            
            # Find the best optimum by starting from n_restart different random points.
            for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
                res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
                if res.fun < min_val:
                    min_val = res.fun[0]
                    min_x = res.x           
                    
            return min_x#.reshape(-1, 1)

        if self.mode == 'EI':
            self.A = expected_improvement(self.XY, self.gpr)
            bounds = np.array([[self.data[:, 0].min(), self.data[:, 0].max()]])
            self.next = propose_location(expected_improvement, self.XY, self.Y, self.gpr, bounds, n_restarts=25)
        else:
            self.A = gp_ucb(self.XY)

            # find the maximal value in the acquisition function
            best = np.argmax(self.A)
            x0 = self.XY[best]

            # find the optimal value from this regressor
            self.res = minimize(gp_ucb, x0)
            self.next = self.res.x

        # calc error for proposed points
        mu = torch.tensor([[self.next[0]]], requires_grad=True, dtype=torch.float64)
        beta = torch.tensor([[self.next[1]]], requires_grad=True, dtype=torch.float64)
        pred_deflection = eng.Eval_Eff_1D_parallel({'mu': mu, 'beta': beta})
        err = self.loss(pred_deflection, eng.target_deflection).detach().numpy()
        
        # adding next point to data
        self.data = np.vstack([self.data, np.array([self.next[0], self.next[1], err])])


        end_time = time.time()
        self.training_time += end_time - start_time

        # adding to global state
        global_memory.gpr_X = self.X
        global_memory.gpr_Y = self.Y
        global_memory.gpr_Z = self.Z
        global_memory.gpr_XY = self.XY

    #TODO: TOTHINK, why  don't we do all these following commands in the evaluate?


    def plot(self, fig_path, global_memory, axis = None):
        if axis:
            axis.set_title('Predicted loss \n GPR')
            axis.contourf(self.X, self.Y, self.Z.reshape(self.X.shape))
            l, = axis.plot(self.generator.E_0, self.generator.nu_0, 'bs')  # white = true value
            l,  = axis.plot(*self.next, 'rs')  # red = predicted value
            return l
            

        fig, ax = plt.subplots(1, 3, figsize=(6, 3))

        ax[0].set_title('Predicted loss')
        ax[0].contourf(self.X, self.Y, self.Z.reshape(self.X.shape))
        ax[0].plot(self.generator.E_0, self.generator.nu_0, 'bs')  # white = true value
        ax[0].plot(*self.next, 'rs')  # red = predicted value

        ax[1].set_title('Uncertainty')
        ax[1].contourf(self.X, self.Y, self.U.reshape(self.X.shape))

        ax[2].set_title('Acquisition function')
        ax[2].contourf(self.X, self.Y, self.A.reshape(self.X.shape))


        plt.savefig(fig_path, dpi = 300)
        plt.close()

    def summary(self, global_memory): # TODO: convert this table to a function..
        Z = -self.gpr.predict(self.XY, return_std=False)
        Z = Z.reshape(self.X.shape)

        mu = np.max(Z[:, 0])
        beta = np.max(Z[:, 1])
        # plot
        E_f, nu_f = youngs_poisson(mu, beta)
        relative_E_error = (E_f* 10**5-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100
        print("\n--------------GPRL---------------")
        print('elapsed time:    {:.2f} (s)'.format(self.training_time))
        print('ground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error,
                                                        relative_nu_error))
        
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])

    def evaluate(self, global_memory):
        pass


class SGD(Model):
    def __init__(self, params, eng, global_memory, model_params = None):
        super().__init__(params)

        self.optimizer = torch.optim.Adam(self.generator.parameters()[:-1], lr=1e5, betas=(params.beta1, params.beta2))
        # self.optimizer = torch.optim.Adagrad(self.generator.parameters()[:-1], lr = 1e5, weight_decay=0)
        self.loss = torch.nn.MSELoss()


    def train(self, eng, t, global_memory):
        """t: is the tqdm; global memory holds states of history and date if needs to be shared across models"""
        start_time = time.time()

        data = self.generator.generate()
        # print('sgd data is ', data)
        err = eng.Eval_Eff_1D_SGD(data)
        # pred_deflection = eng.Eval_Eff_1D_SGD(data)
        
        self.optimizer.zero_grad()
        # err = torch.log(self.loss(pred_deflection, eng.target_deflection))
        err.backward()
        self.optimizer.step()


        self.history = np.vstack([self.history, [data['mu'][0][0].detach().numpy(), data['beta'][0][0].detach().numpy(), err.detach().numpy()]])  
        E_f, nu_f = youngs_poisson(data['mu'].detach().numpy(),
                            data['beta'].detach().numpy())
    
        self.data = np.vstack([self.data, [E_f, nu_f, err.detach().numpy()]])

        end_time = time.time()
        self.training_time += end_time - start_time

        # t.set_description(f"SGD Loss: {err}") #, refresh=True

    def plot(self, fig_path, global_memory, axis = None):
        # fig, ax = plt.subplots(1,2, figsize=(6,3))
        if axis:
            ax = axis
        else:
            fig, ax = plt.subplots(figsize=(6,3))


        # try: ax[0].contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape)) # these are gaussian models' values
        # except: pass
        # ax[0].set_title('history of mu and beta')
        # ax[0].plot(self.history[:, 0], self.history[:, 1], '-x')  # values obtained by torch
        # ax[0].plot(self.generator.E_0, self.generator.nu_0, 'gs')  # white = true value

        # try: if global_memory.gpr_X: ax[1].contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        # except: pass
        # # if : global_memory.gpr_X: ax[1].contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        # ax[1].set_title('history of E_0 and nu_0')
        # ax[1].plot(self.data[:, 0], self.data[:, 1], '-x')  # values obtained by torch
        # ax[1].plot(self.generator.E_0, self.generator.nu_0, 'gs')  # white = true value


        try: ax.contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        except: pass
        # if : global_memory.gpr_X: ax[1].contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        ax.set_title('history of $E$ and $Nu$ \n SGD')
        ax.plot(self.data[:, 0], self.data[:, 1], '-k')  # values obtained by torch
        ax.plot(self.generator.E_0, self.generator.nu_0, 'bs')  # red = true value
        ax.plot(self.data[-1, 0], self.data[-1, 1], 'g.')  # red = true value
        ax.set_xlabel('$E$', fontsize=10)
        ax.set_ylabel('$Nu$', fontsize='medium')

        if axis: return ax
            
        plt.savefig(fig_path, dpi = 300)
        plt.close()


    def evaluate(self, global_memory):
        # print("================SGD=================")
        # print('\nground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        # E_f, nu_f = youngs_poisson(self.generator.mu[0, 0].detach().numpy(),
        #                         self.generator.beta[0, 0].detach().numpy())
        # print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        # print('error:           {:7.2f}% {:7.2f}%'.format((E_f-self.generator.E_0)/self.generator.E_0*100,
        #                                                 (nu_f-self.generator.nu_0)/self.generator.nu_0*100))
        global_memory.sgd_history = self.history
        global_memory.sgd_data = self.data

    def summary(self, global_memory):
        E_f, nu_f = youngs_poisson(self.generator.mu[0, 0].detach().numpy(),
                                self.generator.beta[0, 0].detach().numpy())
        relative_E_error = (E_f-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100

        print("\n-------------SGD---------------")
        print('elapsed time:    {:.2f} (s)'.format(self.training_time))
        print('ground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))

        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error,
                                                        relative_nu_error))
        
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])
        

class GA(Model):
    def __init__(self, params, eng, global_memory, model_params = None):
        super().__init__(params)

        loss = torch.nn.MSELoss()
        def efficiency(data):

            if len(data) > 2: # avg error of runs
                data = [err[0] for err in data]
                return sum(data)/len(data),
            if (data[0] <= 0) or (data[1] <= 0): # penalize invalid values
                return -100000,


            E_f, nu_f = lame(data[0]*1e7, data[1])
            if E_f < 0 or nu_f < 0: # penalize negative numbers
                return -10000,
            E_f_mag = math.floor(math.log10(E_f))
            nu_f_mag = math.floor(math.log10(nu_f))

            # if (E_f_mag != 6) or (nu_f_mag != 6): # penalize magnitude
            #     return -10000,
            
            if (E_f_mag != 6): E_f = E_f * 10**(6 - E_f_mag)
            if (nu_f_mag != 6): nu_f = nu_f * 10**(6 - nu_f_mag)# penalize magnitude
                
            data = {'mu': E_f, 'beta':nu_f}
            # print('data is ', data)

            if (data['mu'] <= 0) or (data['beta'] <= 0): # penalize invalid values
                return -100000,

            result =  torch.log(loss(eng.Eval_Eff_1D_parallel(data), eng.target_deflection)).sum().detach().tolist(),

            return result

        self.creator = creator
        self.creator.create("FitnessMax", base.Fitness, weights=(1.0, .7))
        self.creator.create("Individual", list, fitness=self.creator.FitnessMax)
        IND_SIZE = 2
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.uniform, 0.1, 1) # prior of the population
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                        self.toolbox.attribute, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", efficiency)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        self.MU, self.LAMBDA = 10, 10
        self.pop = self.toolbox.population(n=self.MU)
        self.hof = None

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        # self.stats.register("avg", np.mean, axis=0)
        # self.stats.register("std", np.std, axis=0)
        # self.stats.register("min", np.min, axis=0)
        # self.stats.register("max", np.max, axis=0)
        self.stats.register("efficiency", efficiency)

        # self.count = 0
    def train(self, eng, t, global_memory):
        """t: is the tqdm; global memory holds states of history and date if needs to be shared across models"""        
        pass

        # return pop, logbook, hof

    def plot(self, fig_path, global_memory, axis = None):
        fig, ax = plt.subplots(figsize=(6,3))


        try: ax.contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        except: pass

        ax.set_title('history of E_0 and nu_0')
        # print('pop is ', self.pop)
        ax.plot(self.hof[0][0], self.hof[0][1], 'rs')  # values obtained by torch
        ax.plot(self.generator.E_0, self.generator.nu_0, 'bs')  # white = true value

        plt.savefig(fig_path, dpi = 300)
        plt.close()

    def evaluate(self, global_memory):
        hof = tools.ParetoFront()

        print('\n')
        pop, logbook = algorithms.eaMuPlusLambda(self.pop, self.toolbox, mu=self.MU, lambda_=self.LAMBDA,
                                                cxpb=0.7, mutpb=0.3, ngen=5, 
                                                stats=self.stats, halloffame=hof)
        
        self.pop = pop
        self.hof = hof # HOF gets replaced not updated, fix it so that the best error is added on top!
        # print('log book is ', logbook)

    
        self.data = np.vstack([self.data, [pop, logbook, hof]])
        # print('hof is ', hof)
        print("----------------GA-----------------")
        print('\nground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        E_f, nu_f = lame(self.hof[0][0]*1e7, self.hof[0][1]) 
        E_f, nu_f = youngs_poisson(E_f, nu_f)


        E_f_mag = math.floor(math.log10(E_f))
        nu_f_mag = math.floor(math.log10(nu_f))

        if (E_f_mag != 6): 
            # print('multiplying by ', E_f_mag)
            # print('E_f by ', E_f)
            E_f = E_f * 10**(E_f_mag - 6)

        relative_E_error = (E_f-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100
        # if (nu_f_mag != 6): nu_f = nu_f * 10**(6 - nu_f_mag)# penalize magnitude


        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error, relative_nu_error))
        # print('inverted values: {:.2e} {:.2e}'.format(E_f* 10**E_f_coef, nu_f* 10**nu_f_coef))
        # print('error:           {:7.2f}% {:7.2f}%'.format((E_f* 10**E_f_coef-self.generator.E_0)/self.generator.E_0*100,
        #                                                 (nu_f* 10**nu_f_coef-self.generator.nu_0)/self.generator.nu_0*100))
        print('hof is', hof)
        print("---------------------------------")
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])
        global_memory.ga_history = self.history
        global_memory.ga_data = self.data

    def summary(self, global_memory):
        pass


class PSO(Model):
    def __init__(self, params, eng, global_memory, model_params = None):
        super().__init__(params)

        self.creator = creator
        self.creator.create("FitnessMax", base.Fitness, weights=(1.0, .2))
        self.creator.create("Particle", list, fitness=self.creator.FitnessMax, speed=list)
        # loading from memory:
        # try: self.data = global_memory.gpr_data
        # except: pass
        
        # params
        self.best = None
        self.update_per_run = 30
        
        
        # loss
        loss = nn.MSELoss()
        def efficiency(data):

            if len(data) > 2: # avg error of runs
                data = [err[0] for err in data]
                return sum(data)/len(data),
            if (data[0] <= 0) or (data[1] <= 0): # penalize invalid values
                return -100000,

            if data[0] > 1: data[0] = data[0] / 10
            if data[1] > 1: data[1] = data[1] / 10
            E_f, nu_f = lame(data[0]*1e7, data[1])
            if E_f < 0 or nu_f < 0: # penalize negative numbers
                return -10000,
            E_f_mag = math.floor(math.log10(E_f))
            nu_f_mag = math.floor(math.log10(nu_f))

            # if (E_f_mag != 6) or (nu_f_mag != 6): # penalize magnitude
            #     return -10000,
            
            if (E_f_mag != 6): E_f = E_f * 10**(6 - E_f_mag)
            if (nu_f_mag != 6): nu_f = nu_f * 10**(6 - nu_f_mag)# penalize magnitude
                
            data = {'mu': E_f, 'beta':nu_f}
            # print('data is ', data)

            if (data['mu'] <= 0) or (data['beta'] <= 0): # penalize invalid values
                return -100000,

            # result =  torch.log(loss(eng.Eval_Eff_1D_parallel(data), eng.target_deflection)).sum().detach().tolist(),
            result =  loss(eng.Eval_Eff_1D_parallel(data), eng.target_deflection).sum().detach().tolist(),

            return result


        self.toolbox = base.Toolbox()
        self.toolbox.register("particle", self.generate, size=2, pmin=0.5, pmax=1, smin=0, smax=3)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle, phi1=.1, phi2=1.0)
        self.toolbox.register("evaluate", efficiency)

        self.pop = self.toolbox.population(n=5)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        # self.stats.register("avg", np.mean)
        # self.stats.register("std", np.std)
        # self.stats.register("min", np.min)
        # self.stats.register("max", np.max)
        self.stats.register("efficiency", efficiency)

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "evals"] + self.stats.fields



    @staticmethod
    def generate(size, pmin, pmax, smin, smax):
        part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
        part.speed = [random.uniform(smin, smax) for _ in range(size)]
        part.smin = smin
        part.smax = smax
        part.best = None
        return part


    @staticmethod
    def updateParticle(part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        # part[:] = list(map(operator.add, part, part.speed))

        part[:] = list(map(operator.add, part, part.speed))

    def train(self, eng, t, global_memory):
        global_memory.pso_data = self.data

        # adding to global state
        global_memory.pso_X = self.best
        # print(self.best)


    def plot(self, fig_path, global_memory, axis = None):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        # plot

        plt.savefig(fig_path, dpi = 300)
        plt.close()

    def summary(self, global_memory): # TODO: convert this table to a function..
        # TODO:predict mu and beta

        print("\n--------------PSO---------------")
        print('elapsed time:    {:.2f} (s)'.format(self.training_time))
        print('ground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        # print('data is ', self.data)
        E_f, nu_f = lame(self.data[-1][0]*1e7, self.data[-1][1]) 
        E_f, nu_f = youngs_poisson(E_f, nu_f)


        E_f_mag = math.floor(math.log10(E_f))
        nu_f_mag = math.floor(math.log10(nu_f))

        if (E_f_mag != 6): 
            # print('multiplying by ', E_f_mag)
            # print('E_f by ', E_f)
            E_f = E_f * 10**(E_f_mag - 6)

        relative_E_error = (E_f-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100


        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error, relative_nu_error))
        print("---------------------------------")
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])
        global_memory.pso_history = self.history
        global_memory.pso_data = self.data
    def evaluate(self, global_memory):
        
        start_time = time.time()
        
        #TODO: moved the training process to eval
        # training
        print('\n')
        for g in range(self.update_per_run):
            for part in self.pop:
                part.fitness.values = self.toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not self.best or self.best.fitness < part.fitness:
                    self.best = creator.Particle(part)
                    self.best.fitness.values = part.fitness.values
            for part in self.pop:
                self.toolbox.update(part, self.best)
                # print('best value', self.best)

            # Gather all the fitnesses in one list and print the stats
            self.logbook.record(gen=g, evals=len(self.pop), **self.stats.compile(self.pop))
            # print(self.logbook.stream)
    
        # print('pop is', self.pop)
        print('log books', self.logbook)
        print('PSO best is ', self.best)    
        # adding next point to data
        self.data = np.vstack([self.data, np.array([self.best[0], self.best[1], self.logbook])])


        end_time = time.time()
        self.training_time += end_time - start_time


class PSOL(Model):

    # pso_1 = PSO(particles.copy(), velocities.copy(), fitness_function, w=0.1, c_1=4.0, c_2=0.0, auto_coef=False)

    def __init__(self, params, eng, global_memory, model_params = None, particles = None, velocities = None, fitness_function = None,
                 w=0.8, c_1=1, c_2=1, max_iter=100, auto_coef=True):
        super().__init__(params)

        self.folder = params.output_dir +  f'/figures/PSOL/'
        self.gif_folder = params.output_dir + f'/figures/deviceSamples/'

        n_particles = 10

        # self.E_r = self.E_0 / 4 * np.random.randn() + self.E_0
        # self.nu_r = self.nu_0 / 4 * np.random.randn() + self.nu_0
        self.particles = np.random.uniform(0.1, 1, (n_particles, 2))
        velocities = (np.random.random((n_particles, 2)) - 0.5) / 10
        self.velocities = velocities

        loss = nn.MSELoss()
        def fitness_function(data):
            result = []
            for i in data:
                if i[0] < 0 or i[1] <  0:
                    result.append(10000)
                    continue
                if i[0] > 1:
                    i[0] /= 10
                if i[1] > 1:
                    i[1] /= 10
                E_f, nu_f = lame(i[0]*1e7, i[1])
                data = {'mu': E_f, 'beta': nu_f}
                # print('data is', data)
                result.append(loss(eng.Eval_Eff_1D_parallel(data), eng.target_deflection).sum().detach().tolist())

            # print(result)
            return result

        self.fitness_function = fitness_function

        self.N = len(self.particles)
        self.w = w
        self.c_1 = c_1
        self.c_2 = c_2
        self.auto_coef = auto_coef
        self.max_iter = max_iter


        self.p_bests = self.particles
        start_time = time.time()

        self.p_bests_values = self.fitness_function(self.particles)

        
        self.g_best = self.p_bests[0]
        self.g_best_value = self.p_bests_values[0]
        self.update_bests()

        self.iter = 0
        self.is_running = True
        self.update_coef()
        end_time = time.time()
        self.training_time += end_time - start_time

    def __str__(self):
        return f'[{self.iter}/{self.max_iter}] $w$:{self.w:.3f} - $c_1$:{self.c_1:.3f} - $c_2$:{self.c_2:.3f}'

    def train(self, eng, t, global_memory):
        pass

    def update_coef(self):
        if self.auto_coef:
            t = self.iter
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.c_1 = -3 * t / n + 3.5
            self.c_2 =  3 * t / n + 0.5

    def move_particles(self):

        # add inertia
        new_velocities = self.w * self.velocities
        # add cognitive component
        r_1 = np.random.random(self.N)
        r_1 = np.tile(r_1[:, None], (1, 2))
        new_velocities += self.c_1 * r_1 * (self.p_bests - self.particles)
        # add social component
        r_2 = np.random.random(self.N)
        r_2 = np.tile(r_2[:, None], (1, 2))
        g_best = np.tile(self.g_best[None], (self.N, 1))
        new_velocities += self.c_2 * r_2 * (g_best  - self.particles)

        self.is_running = np.sum(self.velocities - new_velocities) != 0

        # update positions and velocities
        self.velocities = new_velocities
        self.particles = self.particles + new_velocities


    def update_bests(self):
        fits = self.fitness_function(self.particles)

        for i in range(len(self.particles)):
            # update best personnal value (cognitive)
            if fits[i] < self.p_bests_values[i]:
                self.p_bests_values[i] = fits[i]
                self.p_bests[i] = self.particles[i]
                # update best global value (social)
                if fits[i] < self.g_best_value:
                    self.g_best_value = fits[i]
                    self.g_best = self.particles[i]

    def evaluate(self, global_memory):
        start_time = time.time()

        if self.iter > 0:
            self.move_particles()
            self.update_bests()
            self.update_coef()

        self.iter += 1
        self.is_running = self.is_running and self.iter < self.max_iter
        
        end_time = time.time()
        self.training_time += end_time - start_time
        self.quick_save_fig(self.folder + f'{self.iter}.png')

        return self.is_running


    def summary(self, global_memory):
        # TODO:predict mu and beta

        saving_folder = self.gif_folder + f'/{self.iter}_tmp.gif'
        make_gif_from_folder(self.folder, saving_folder)

        print("\n--------------PSOL---------------")
        print('elapsed time:    {:.2f} (s)'.format(self.training_time))
        print('ground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        E_f, nu_f = lame(self.g_best[0]*1e7, self.g_best[1]) 
        E_f, nu_f = youngs_poisson(E_f, nu_f)


        E_f_mag = math.floor(math.log10(E_f))
        nu_f_mag = math.floor(math.log10(nu_f))

        if (E_f_mag != 6): 
            E_f = E_f * 10**(E_f_mag - 6)

        relative_E_error = (E_f-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100


        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error, relative_nu_error))
        print("---------------------------------")
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])
        global_memory.psol_history = self.history
        global_memory.psol_data = self.data





    def plot(self, fig_path, global_memory, axis = None):    
        normalize = True

        if axis:
            axis.set_title('Predicted loss \n PSOL')
            try: ax.contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
            except: pass
            E_f, nu_f = lame(self.g_best[0]*1e7, self.g_best[1]) 
            E_f, nu_f = youngs_poisson(E_f, nu_f)
            l, = axis.plot(self.generator.E_0, self.generator.nu_0, 'bs')  # blue = true value
            l,  = axis.plot(E_f, nu_f, 'rs')  # red = predicted value
            return l
            

        fig, ax = plt.subplots(1, 2, figsize=(6, 3))

        E_f, nu_f = lame(self.g_best[0]*1e7, self.g_best[1]) 
        E_f, nu_f = youngs_poisson(E_f, nu_f)

        ax[0].set_title('Predicted loss \nPSOL')
        try: ax.contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        except: pass
        ax[0].plot(self.generator.E_0, self.generator.nu_0, 'bs')  # white = true value
        ax[0].plot(E_f, nu_f, 'rs')  # red = predicted value

        # plot contour
        ax[1].set_title('All Particles \nPSOL')
        try: ax.contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        except: pass
        X, Y = self.particles.swapaxes(0, 1)
        X, Y = lame(X*1e7, Y)
        X, Y = youngs_poisson(X, Y)
        if self.velocities is not None:
            U, V = self.velocities.swapaxes(0, 1)
            if normalize:
                N = np.sqrt(U**2+V**2)
                U, V = U/N, V/N
        ax[1].scatter(X, Y, color='#000')
        if self.velocities is not None:
            ax[1].quiver(X, Y, U, V, color='#000', headwidth=2, headlength=2, width=5e-3)


        plt.savefig(fig_path, dpi = 300)
        plt.close()

    def quick_save_fig(self, fig_path):
        fig, ax = plt.subplots(figsize=(6, 3))
        normalize = True
        # plot contour
        ax.set_title('All Particles \nPSOL')

        X, Y = self.particles.swapaxes(0, 1)
        X, Y = lame(X*1e7, Y)
        X, Y = youngs_poisson(X, Y)
        if self.velocities is not None:
            U, V = self.velocities.swapaxes(0, 1)
            if normalize:
                N = np.sqrt(U**2+V**2)
                U, V = U/N, V/N
        ax.scatter(X, Y, color='#000')
        if self.velocities is not None:
            ax.quiver(X, Y, U, V, color='#000', headwidth=2, headlength=2, width=5e-3)


        plt.savefig(fig_path, dpi = 300)
        plt.close()