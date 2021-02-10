import torch
import logging
from dolfin import vertex_to_dof_map

class engine:
    def __init__(self, fenics_model, batch_size, mu, beta, force):
        #TODO: our [mu, beta, force] itself must differ; something that currently is not happening
        self.model = fenics_model
        self.v2d = vertex_to_dof_map(self.model.V)
        self.batch_size = batch_size
        self.mu = mu
        self.beta = beta
        self.force = force

    def Eval_Eff_1D_parallel(self, img, mu, beta, force):
        mu = torch.normal(mean=self.mu, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        beta = torch.normal(mean=self.beta, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        force = torch.normal(mean=self.force, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)

        self.u = self.model(mu, beta, force)

        difference = self.u.flatten()[self.v2d].reshape(-1, 3).unsqueeze_(0).repeat(10, 1, 1) - img

        return difference.detach().float()
    
    def GradientFromSolver_1D_parallel(self, img):
        effs_and_gradients = []
        
        mu = torch.normal(mean=self.mu, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        beta = torch.normal(mean=self.beta, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        force = torch.normal(mean=self.force, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)

        # compute gradients for all!
        self.u = self.model(mu, beta, force)
        
        # v1.2
        if self.batch_size == 1:
            difference = self.u.flatten()[self.v2d].reshape(-1, 3).unsqueeze_(0).repeat(10, 1, 1)
        else:
            difference = torch.zeros((self.batch_size, 176, 3))
            for i in range(self.batch_size):
                diffs = self.u[i].flatten()[self.v2d].reshape(-1, 3).unsqueeze(0) # what's a more efficient way?
                difference[i, :, :] = diffs 


        # v1.1
        # if self.batch_size == 1:
        #     difference = self.u.flatten()[self.v2d].reshape(-1, 3).unsqueeze_(0).repeat(10, 1, 1) - img
        # else:
        #     difference = torch.zeros((self.batch_size, 176, 3))
        #     for i in range(self.batch_size):
        #         diffs = self.u[i].flatten()[self.v2d].reshape(-1, 3).unsqueeze(0) - img[i] # what's a more efficient way?
        #         difference[i, :, :] = diffs

        

        effs_and_gradients.append(difference)

        J = torch.sum(torch.mean(difference, dim=0).view(-1)).backward()

        dJdmu = mu.grad
        dJbeta = beta.grad
        # dJforce = force.grad

        
        effs_and_gradients.append(dJdmu.detach().numpy()) # since we have to revert it back to tensor
        effs_and_gradients.append(dJbeta.detach().numpy()) # since we have to revert it back to tensor
        # effs_and_gradients.append(force.grad.detach().numpy()) # since we have to revert it back to tensor
        J = None
        
        return effs_and_gradients
