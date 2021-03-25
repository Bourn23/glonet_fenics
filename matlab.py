import torch
import logging
from dolfin import vertex_to_dof_map
from utils import lame, youngs_poisson
import math
import random

class engine:
    def __init__(self, fenics_model, batch_size, E_0, nu_0, force):
        self.SGD = False
        #TODO: our [mu, beta, force] itself must differ; something that currently is not happening
        self.model = fenics_model
        self.v2d = vertex_to_dof_map(self.model.V)
        self.batch_size = batch_size
        mu, beta = lame(E_0, nu_0)
        self.mu = torch.tensor([[mu]], requires_grad=True, dtype=torch.float64)
        self.beta = torch.tensor([[beta]], requires_grad=True, dtype=torch.float64)
        self.force = torch.tensor([[force]], requires_grad=True, dtype=torch.float64)
        self.target_deflection = self.model(self.mu, self.beta, self.force).detach() # sure?


        self.loss = torch.nn.MSELoss()
        
    def Eval_Eff_1D_SGD(self, data):
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
                mu = torch.tensor([[data[0]* 10**mu_coef]] * self.batch_size, requires_grad=True, dtype=torch.float64)
                beta = torch.tensor([[data[1]* 10**beta_coef]] * self.batch_size, requires_grad=True, dtype=torch.float64)
                force = self.force


        u = self.model(mu, beta, force)
        # print(u)

       
        if self.SGD:
            # v2.
            start = random.randint(0, 175) # keep # smples fixed
            random_elements = random.randint(0, 175)
            if random_section < random_elements:
                err = torch.log(self.loss(u[0, random_section:random_elements, :], self.target_deflection[0, random_section:random_elements, :]))
                return err
            else: 
                err = torch.log(self.loss(u[0, random_elements:random_section, :], self.target_deflection[0, random_elements:random_section, :]))
                return err
            # v1. note its only one element!
            # random_elements = random.randint(0, 175)
            # return u[:, :random_elements]
            err = torch.log(self.loss(u[:, random_elements, :], self.target_deflection[:, random_elements, :]))
            return err

        else:
            err = torch.log(self.loss(u, self.target_deflection))
            return err


    def Eval_Eff_1D_parallel(self, data):
        
        # if self.batch_size != 1: # chnged == with !=
        #     mu = torch.tensor([[data['mu']]] * self.batch_size, requires_grad=True, dtype=torch.float64)
        #     beta = torch.tensor([[data['beta']]] * self.batch_size, requires_grad=True, dtype=torch.float64)
        #     force = torch.tensor([[data['force']]], requires_grad=True, dtype=torch.float64)

        # else:  
        #     try:
        #         # print('SGDmu is ', data['mu'])
        #         # print('SGDbeta is ', data['beta'])
        #         mu = data['mu']
        #         beta = data['beta']
        #         force = self.force
        #     except: # for handling the GA data
        #         if data[0] < 0: data[0] = abs(data[0])
        #         elif data[0] == 0: data[0] = 1e-7
        #         if data[1] <= 1e-8: data[1] = 1e-8#abs(data[1])

        #         # print('GA mu is ', data[0]*1e7)
        #         # print('GA beta is ', data[1]*1e8)
        #         mu_coef =  math.floor(math.log(1e7, 10) - math.log(data[0], 10))
        #         beta_coef = math.floor(math.log(1e8, 10) - math.log(data[1], 10))
        #         mu = torch.tensor([[data[0]* 10**mu_coef]] * self.batch_size, requires_grad=True, dtype=torch.float64)
        #         beta = torch.tensor([[data[1]* 10**beta_coef]] * self.batch_size, requires_grad=True, dtype=torch.float64)
        #         force = self.force

        mu = torch.tensor([[data['mu']]] * self.batch_size, requires_grad=True, dtype=torch.float64)
        beta = torch.tensor([[data['beta']]] * self.batch_size, requires_grad=True, dtype=torch.float64)
        try: force = torch.tensor([[data['force']]], requires_grad=True, dtype=torch.float64)
        except: force = self.force

        # print(f'mu is {mu} and beta is {beta}')

        u = self.model(mu, beta, force)
        print(u)
        if self.SGD:
            # v2.
            # random_section = random.randint(0, 175)
            # random_elements = random.randint(0, 175)
            # if random_section < random_elements:
            #     return u[:, random_section:random_elements]
            # else: return u[:, random_elements:random_section]
            # v1.
            random_elements = random.randint(0, 175)
            return u[:, :random_elements]

        else:
            return u
    
    def GradientFromSolver_1D_parallel(self, data):
        
        # why do you want to keep it this way? based on the existing values, it generates multiple variants of it so can we use those values for faster convergence?
        # again something like a global optimizer
        
        mu = torch.tensor(data['mu'], requires_grad=True, dtype=torch.float64).view(-1, 1)
        beta = torch.tensor(data['beta'], requires_grad=True, dtype=torch.float64).view(-1, 1)

        try: force = torch.tensor([[data['force']]], requires_grad=True, dtype=torch.float64)
        except: 
            if mu.shape[0] == 1:
                force = self.force
            else:
                force = self.force.expand(mu.shape[0], 1)
        # print(f'mu shape {mu.shape}, beta {beta.shape}, force {force.shape}')
        self.u = self.model(mu, beta, force)
        loss = torch.nn.MSELoss()
        # print('size of u ', self.u.shape) # 441, 176, 3

        # v1.3
        output = torch.zeros((1, 1))
        if self.u.shape[0] == 1:
            output = loss(self.u, self.target_deflection)
        else:
            # target = self.target_deflection.expand(self.u.shape[0], 176, 3) # expected 441, 176, 3
            # declare a loss function
            # apply error over function
            #
            for i in range(self.u.shape[0]):       
                output = torch.vstack([output, loss(self.u[i, :, :], self.target_deflection).detach().unsqueeze(0)])

            # print('output in the loss ', output.shape) # expected 441, 176, 3 -> 441, 1
            # output = torch.mean(torch.mean(output, dim=2), dim=1).detach()#.sum()
            # output = output.sum(axis = 0) # expected 441, 1
            # print('output after summation size:', output.size())
            # output = output[1:,:].expand(mu.shape[0], 2).numpy()
            a_ = torch.tensor([i for i in range(mu.shape[0]) if i % 2 == 0])
            b_ = torch.tensor([i for i in range(mu.shape[0]) if i % 2 == 1])
            inds1 = torch.vstack([a_.view(-1, 2), b_.view(-1, 2)])
            print('inds1 are ', inds1.shape)
            b = torch.tensor([1,2]).repeat(data['mu'].shape[0]//2)
            print('b shape is ', b.shape)
            _, inds2 = torch.sort(b)
            output = output[1:,:].flatten()
            print('final0 flatten shape: ', output.shape)
            output = output[1:,:].flatten()[inds1]#
            print('final1 inds1 shape: ', output.shape)
            output = output[inds2].numpy() # 441,1
            print('final2 inds2 shape: ', output.shape)

            
            # .view(data['mu'].shape[0], -1).numpy()
            # print('output after expansion', output.shape)

        # effs_and_gradients.append([1])
        
        return output
