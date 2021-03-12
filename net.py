import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 
import matplotlib.pyplot as plt
from utils import lame, youngs_poisson

# GPR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from scipy.optimize import minimize
from sklearn.model_selection import GridSearchCV


# GA
from deap import base, creator
from deap import tools
import random


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
        self.loss_history = np.zeros([0])
        self.data    = np.zeros([0,3])

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

    def plot(self, fig_path, global_memory):
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
        self.init_data(eng, 200)


    def init_data(self, eng, i  = 200):
        for i in range(i):
            parameters = self.generator.generate(sampling = True)

            pred_deflection = eng.Eval_Eff_1D_parallel(parameters)
            err = self.loss(pred_deflection, eng.target_deflection).detach().numpy()
            
            # build internal memory
            self.data = np.vstack([self.data, np.array([parameters['E_r'], parameters['nu_r'], err])])

    # @staticmethod
    

    def train(self, eng, t, global_memory):
        self.init_data(eng, 1)
    #TODO: TOTHINK, why  don't we do all these following commands in the evaluate?


    def plot(self, fig_path, global_memory):
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))

        ax[0].set_title('Predicted loss')
        ax[0].contourf(self.X, self.Y, self.Z.reshape(self.X.shape))
        ax[0].plot(self.generator.E_0, self.generator.nu_0, 'ws')  # white = true value
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
        print("\n--------------GPR---------------")
        print('\nground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        E_f, nu_f = youngs_poisson(mu,
                                beta)
        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        relative_E_error = (E_f* 10**5-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100
        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error,
                                                        relative_nu_error))
        
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])

    def evaluate(self, global_memory):
        global_memory.gpr_data = self.data
        #TODO: moved the training process to eval
        ls = np.std(self.data, axis=0)[:2]

        # HyperParam Opt.
        param_grid = [{
                "alpha":  [0, 1],
                "kernel": [RBF(ls)]
            }, {
                "alpha":  [0, 1],
                "kernel": [DotProduct(sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
            }, {
                "alpha":  [0, 1],
                "kernel": [WhiteKernel(noise_level = sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
            }, {
                "alpha":  [0, 1],
                "kernel": [DotProduct() + WhiteKernel(noise_level = sigma_0) for sigma_0 in np.logspace(-1, 1, 2)]
            }, {
                "alpha":  [0, 1],
                'kernel' : [1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))]
            }]
            
            
            
        scores = ['explained_variance', 'r2']
        
        kernel = DotProduct() + WhiteKernel() + RBF(ls)
        self.gpr = GaussianProcessRegressor()

        for score in scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(estimator=self.gpr, param_grid=param_grid, cv=4,
                            scoring='%s' % score)
            clf.fit(self.data[:, :2], np.log(self.data[:, 2]))
            print('best param : ',clf.best_params_)


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

        
        self.A = gp_ucb(self.XY)

        # find the maximal value in the acquisition function
        best = np.argmax(self.A)
        x0 = self.XY[best]

        # find the optimal value from this regressor
        self.res = minimize(gp_ucb, x0)

        self.next = self.res.x

        # adding to global state
        global_memory.gpr_X = self.X
        global_memory.gpr_Y = self.Y
        global_memory.gpr_Z = self.Z
        global_memory.gpr_XY = self.XY


class SGD(Model):
    def __init__(self, params, eng, global_memory, model_params = None):
        super().__init__(params)
        self.optimizer = torch.optim.Adam(self.generator.parameters()[:-1], lr=params.lr, betas=(params.beta1, params.beta2))
        self.loss = torch.nn.MSELoss()

    def train(self, eng, t, global_memory):
        """t: is the tqdm; global memory holds states of history and date if needs to be shared across models"""
        data = self.generator.generate()
        pred_deflection = eng.Eval_Eff_1D_parallel(data)

        self.optimizer.zero_grad()
        err = torch.log(self.loss(pred_deflection, eng.target_deflection))
        err.backward()
        self.optimizer.step()

        self.history = np.vstack([self.history, [data['mu'][0][0].detach().numpy(), data['beta'][0][0].detach().numpy(), err.detach().numpy()]])  
        E_f, nu_f = youngs_poisson(data['mu'].detach().numpy(),
                            data['mu'].detach().numpy())
    
        self.data = np.vstack([self.data, [E_f, nu_f, err.detach().numpy()]])


        # t.set_description(f"SGD Loss: {err}") #, refresh=True

    def plot(self, fig_path, global_memory):
        # fig, ax = plt.subplots(1,2, figsize=(6,3))
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
        ax.set_title('history of E_0 and nu_0')
        ax.plot(self.data[:, 0], self.data[:, 1], '-k')  # values obtained by torch
        ax.plot(self.generator.E_0, self.generator.nu_0, 'rs')  # white = true value

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
        global_memory.sgd_histry = self.history
        global_memory.sgd_data = self.data

    def summary(self, global_memory):
        print("\n-------------SGD---------------")
        print('ground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        E_f, nu_f = youngs_poisson(self.generator.mu[0, 0].detach().numpy(),
                                self.generator.beta[0, 0].detach().numpy())
        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        relative_E_error = (E_f-self.generator.E_0)/self.generator.E_0*100
        relative_nu_error = (nu_f-self.generator.nu_0)/self.generator.nu_0*100
        # print('error:           {:7.2f}% {:7.2f}%'.format((E_f-self.generator.E_0)/self.generator.E_0*100,
        #                                                 (nu_f-self.generator.nu_0)/self.generator.nu_0*100))
        print('error:           {:7.2f}% {:7.2f}%'.format(relative_E_error,
                                                        relative_nu_error))
        
        self.loss_history = np.vstack([self.loss_history, [relative_E_error, relative_nu_error]])
        

class GA(Model):
    def __init__(self, params, eng, global_memory, model_params = None):
        super().__init__(params)
        loss = torch.nn.MSELoss()
        def efficiency(data):
            return torch.log(loss(eng.Eval_Eff_1D_parallel(data), eng.target_deflection)).sum()

        self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self.creator.create("Individual", list, fitness=creator.FitnessMin)
        IND_SIZE = 2

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.random) # prior of the population
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                        toolbox.attribute, n=IND_SIZE)
        self.toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean, axis=0)
        self.stats.register("std", np.std, axis=0)
        self.stats.register("min", np.min, axis=0)
        self.stats.register("max", np.max, axis=0)
        self.stats.register("efficiency", efficiency)

    def train(self, eng, t, global_memory):
        """t: is the tqdm; global memory holds states of history and date if needs to be shared across models"""
        MU, LAMBDA = 100, 200
        data = self.generator.generate()
        pop = self.toolbox.population(n=MU)

        #algo here

        
        
        hof = tools.ParetoFront()

        
        pop, logbook = algorithms.eaMuPlusLambda(pop, self.toolbox, mu=MU, lambda_=LAMBDA,
                                                cxpb=0.7, mutpb=0.3, ngen=40, 
                                                stats=self.stats, halloffame=hof)
        
        

        # self.history = np.vstack([self.history, [data['mu'][0][0].detach().numpy(), data['beta'][0][0].detach().numpy(), err.detach().numpy()]])  
        # E_f, nu_f = youngs_poisson(data['mu'].detach().numpy(),
        #                     data['mu'].detach().numpy())
    
        self.data = np.vstack([self.data, [pop, logbook, hof]])

        # return pop, logbook, hof

    def plot(self, fig_path, global_memory):
        fig, ax = plt.subplots(figsize=(6,3))


        try: ax.contourf(global_memory.gpr_X, global_memory.gpr_Y, global_memory.gpr_Z.reshape(global_memory.gpr_X.shape))
        except: pass

        ax.set_title('history of E_0 and nu_0')
        ax.plot(self.data[:, 0], self.data[:, 1], '-k')  # values obtained by torch
        ax.plot(self.generator.E_0, self.generator.nu_0, 'rs')  # white = true value

        plt.savefig(fig_path, dpi = 300)
        plt.close()

    def evaluate(self, global_memory):
        print('\nground truth:    {:.2e} {:.2e}'.format(self.generator.E_0, self.generator.nu_0))

        E_f, nu_f = youngs_poisson(self.generator.mu[0, 0].detach().numpy(),
                                self.generator.beta[0, 0].detach().numpy())
        print('inverted values: {:.2e} {:.2e}'.format(E_f, nu_f))
        print('error:           {:7.2f}% {:7.2f}%'.format((E_f-self.generator.E_0)/self.generator.E_0*100,
                                                        (nu_f-self.generator.nu_0)/self.generator.nu_0*100))
        print("---------------------------------")
        global_memory.sgd_histry = self.history
        global_memory.sgd_data = self.data