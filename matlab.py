import torch
import logging
from dolfin import vertex_to_dof_map
import math
import random

class engine:
    def __init__(self, fenics_model, batch_size, density_field):
        self.SGD = False
        #TODO: our [mu, beta, force] itself must differ; something that currently is not happening
        self.model = fenics_model
        # coordinates for plotting density
        self.v2d = self.model.V0.tabulate_dof_coordinates()
        self.batch_size = batch_size
        self.density_field = density_field
        self.rslt = self.model(self.density_field)
        # self.target_densityfield, self.target_deflection = self.rslt[0].detach(), self.rslt[1].detach()
        self.target_deflection = self.rslt.detach()
        self.loss_deflection = torch.nn.MSELoss()
        self.loss_density = torch.nn.MSELoss()
        
    def Eval_Eff_1D_SGD(self, data):
        theta = data.reshape(1, -1)

        # theta_p, \
        u = self.model(theta)
        # print('theta p', theta_p)
        print('displacement', u)
        deflection_error = torch.log(self.loss_deflection(u, self.target_deflection))
        # density_error = torch.log(self.loss_density(theta_p, self.target_densityfield))
        # return density_error, deflection_error
        return deflection_error

    def Eval_Eff_1D_parallel(self, data):
        try: theta = torch.tensor([[data['theta']]], requires_grad=True, dtype=torch.float64)
        except: theta = self.density_field


        theta_p, u = self.model(theta)

        return theta_p, u
    
    def GradientFromSolver_1D_parallel(self, data):
        
        # why do you want to keep it this way? based on the existing values, it generates multiple variants of it so can we use those values for faster convergence?
        # again something like a global optimizer

        try:    theta = torch.tensor([[data['theta']]], requires_grad=True, dtype=torch.float64)
        except: theta = self.density_field

        theta_p, u = self.model(theta)

        return theta_p, u
