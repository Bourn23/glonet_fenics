import torch
import logging
from dolfin import vertex_to_dof_map

class engine:
    def __init__(self, fenics_model, batch_size, mu, beta, force):
        #TODO: our [mu, beta, force] itself must differ; something that currently is not happening
        self.model = fenics_model
        self.v2d = vertex_to_dof_map(self.model.V)
        self.batch_size = batch_size
        self.mu = torch.tensor([[mu]], requires_grad=True, dtype=torch.float64)
        self.beta = torch.tensor([[beta]], requires_grad=True, dtype=torch.float64)
        self.force = torch.tensor([[force]], requires_grad=True, dtype=torch.float64)
        self.target_deflection = self.model(self.mu, self.beta, self.force).detach() # sure?
        self.target_deflection = self.target_deflection.flatten()[self.v2d].reshape(-1, 3)#.unsqueeze_(0).repeat(10, 1, 1)

    def Eval_Eff_1D_parallel(self, img, mu, beta, force):
        mu = torch.normal(mean=self.mu, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        beta = torch.normal(mean=self.beta, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        force = torch.normal(mean=self.force, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)

        self.u = self.model(mu, beta, force)

        difference = self.u.flatten()[self.v2d].reshape(-1, 3).unsqueeze_(0).repeat(10, 1, 1) - img

        return difference.detach().float()
    
    def GradientFromSolver_1D_parallel(self, img):
        effs_and_gradients = []
        
        # why do you want to keep it this way? based on the existing values, it generates multiple variants of it so can we use those values for faster convergence?
        # again something like a global optimizer
        mu = torch.normal(mean=img[0], std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        beta = torch.normal(mean=img[1], std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        force = torch.normal(mean=img[2], std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)

        # compute gradients for all!
        self.u = self.model(mu, beta, force)
        loss = torch.nn.MSELoss()
        # v1.2
        if self.batch_size == 1:
            difference = self.u.flatten()[self.v2d].reshape(-1, 3)#.unsqueeze_(0).repeat(self.batch_size, 1, 1)
            logging.info(f"Differnc shape is {difference.size()}")
            logging.info(f"target_shape shape is {self.target_deflection.size()}")
            output = loss(difference, self.target_deflection)
        else: # make sure value assignment works fine.
            difference = torch.zeros((self.batch_size, 176, 3))
            for i in range(self.batch_size):
                diffs = self.u[i].flatten()[self.v2d].reshape(-1, 3).unsqueeze(0) # what's a more efficient way?
                difference[i, :, :] = diffs 
        

        effs_and_gradients.append(difference.detach())

        J = torch.sum(torch.mean(difference, dim=0).view(-1)).backward()

        dJdmu = mu.grad
        dJbeta = beta.grad
        # dJforce = force.grad

        
        effs_and_gradients.append(dJdmu.detach().numpy()) # since we have to revert it back to tensor
        effs_and_gradients.append(dJbeta.detach().numpy()) # since we have to revert it back to tensor
        # effs_and_gradients.append(force.grad.detach().numpy()) # since we have to revert it back to tensor
        J = None
        
        logging.info("effs_and_gradients")
        logging.info(effs_and_gradients)
        return effs_and_gradients, outputs
