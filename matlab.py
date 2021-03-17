import torch
import logging
from dolfin import vertex_to_dof_map
from utils import lame, youngs_poisson

class engine:
    def __init__(self, fenics_model, batch_size, E_0, nu_0, force):
        #TODO: our [mu, beta, force] itself must differ; something that currently is not happening
        self.model = fenics_model
        self.v2d = vertex_to_dof_map(self.model.V)
        self.batch_size = batch_size
        mu, beta = lame(E_0, nu_0)
        self.mu = torch.tensor([[mu]], requires_grad=True, dtype=torch.float64)
        self.beta = torch.tensor([[beta]], requires_grad=True, dtype=torch.float64)
        self.force = torch.tensor([[force]], requires_grad=True, dtype=torch.float64)
        self.target_deflection = self.model(self.mu, self.beta, self.force).detach() # sure?
        

    def Eval_Eff_1D_parallel(self, data):
        # print('data is ', len(data))
        # if type(data) != dict: return data
        if self.batch_size != 1: # chnged == with !=
            mu = torch.tensor([[data['mu']]] * self.batch_size, requires_grad=True, dtype=torch.float64)
            beta = torch.tensor([[data['beta']]] * self.batch_size, requires_grad=True, dtype=torch.float64)
            force = torch.tensor([[data['force']]], requires_grad=True, dtype=torch.float64)

        else:  
            try:
                # print('SGDmu is ', data['mu'])
                # print('SGDbeta is ', data['beta'])
                mu = data['mu']
                beta = data['beta']
                force = self.force
            except: # for handling the GA data
                if data[0] < 0: data[0] = abs(data[0])
                elif data[0] == 0: data[0] = 1e-7
                if data[1] <= 1e-8: data[1] = 1e-8#abs(data[1])

                # print('GA mu is ', data[0]*1e7)
                # print('GA beta is ', data[1]*1e8)
                mu_coef =  math.floor(math.log(1e7, 10) - math.log(data[0], 10))
                beta_coef = math.floor(math.log(1e8, 10) - math.log(data[1], 10))
                mu = torch.tensor([[data[0]*mu_coef]] * self.batch_size, requires_grad=True, dtype=torch.float64)
                beta = torch.tensor([[data[1]*beta_coef]] * self.batch_size, requires_grad=True, dtype=torch.float64)
                force = self.force


        u = self.model(mu, beta, force)

        return u
    
    def GradientFromSolver_1D_parallel(self, data):
        effs_and_gradients = []
        
        # why do you want to keep it this way? based on the existing values, it generates multiple variants of it so can we use those values for faster convergence?
        # again something like a global optimizer
        mu = data['mu']
        beta = data['beta']
        force = data['force']

        self.u = self.model(mu, beta, force)
        loss = torch.nn.MSELoss()

        # v1.3
        if self.u.shape[0] == 1:
            output = loss(self.u, self.target_deflection)
        else:
            output = loss(self.u, self.target_deflection.repeat(self.u.shape[0], 1, 1))
        
        effs_and_gradients.append([1])
        
        return effs_and_gradients, output
