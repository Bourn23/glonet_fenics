import torch
import logging
from dolfin import vertex_to_dof_map

class engine:
    def __init__(self, fenics_model, batch_size, mu, beta, force):
        #TODO: our [mu, beta, force] itself must differ; something that currently is not happening
        self.model = fenics_model
        self.v2d = vertex_to_dof_map(self.model.V)
        # self.batch_size = batch_size
        self.batch_size = batch_size
        self.mu = torch.normal(mean=mu, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        self.beta = torch.normal(mean=beta, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        self.force = torch.normal(mean=force, std=torch.arange(1, 0, -((1.-0.) / self.batch_size))).type(torch.float64).unsqueeze(1).requires_grad_(True)
        # self.force = torch.tensor([[force]] * self.batch_size, requires_grad = True, dtype = torch.float64)
        self.u = None
        # logging.info(f"matlab_All initialized: type of wave {type(self.mu)} and angle {type(self.beta)}")
    def Eval_Eff_1D_parallel(self, img, mu, beta, force):
        # logging.info("matlab_eval_eff_called")
        # this only works a single image
        if type(self.mu) != torch.Tensor:
            # logging.info("matlab_EVAL_EFF_data_U_Unknown")
            self.mu = torch.tensor([[self.mu]] * self.batch_size, dtype = torch.float64, requires_grad=True)
            self.beta = torch.tensor([[self.beta]] * self.batch_size, dtype = torch.float64, requires_grad=True)
        self.u = self.model(self.mu, self.beta, self.force)
        
        # u_ = self.u.detach().flatten()[self.v2d].reshape(-1, 3)
        # if self.batch_size == 1:
        difference = self.u.flatten()[self.v2d].reshape(-1, 3).unsqueeze_(0).repeat(10, 1, 1) - img
        # else:
        #     difference = torch.zeros((self.batch_size, 176, 3))
        #     for i in range(self.batch_size):
        #         diffs = self.u[i].flatten()[self.v2d].reshape(-1, 3).unsqueeze(0) - img[i] # what's a more efficient way?
        #         difference[i, :, :] = diffs
        #         logging.info(f"diffs shape is {diffs.shape}")


        return difference.float()
    
    def GradientFromSolver_1D_parallel(self, img):
        # What should be going on here? 1. see paper for what they're doing / 2. see the .mat file
        # And what is going on now?:))
        # logging.info("matlab_grad_solver_called")
        # if self.u is None:
        if type(self.mu) != torch.Tensor:
            # logging.info("matlab_EVAL_EFF_data_U_Unknown_TENSORS")
            self.mu = torch.tensor([[self.mu]] * self.batch_size, dtype = torch.float64, requires_grad=True)
            self.beta = torch.tensor([[self.beta]] * self.batch_size, dtype = torch.float64, requires_grad=True)

        # compute gradients for all!
        self.u = self.model(self.mu, self.beta, self.force)
        self.u = self.u.detach()

            # self.u.mean().backward() # mean(axis = 0) to average over batches I'm thinking how to calculate gradients for each and one of them
        if self.batch_size == 1:
            difference = self.u.flatten()[self.v2d].reshape(-1, 3).unsqueeze_(0).repeat(10, 1, 1) - img
        else:
            difference = torch.zeros((self.batch_size, 176, 3))
            for i in range(self.batch_size):
                diffs = self.u[i].flatten()[self.v2d].reshape(-1, 3).unsqueeze(0) - img[i] # what's a more efficient way?
                difference[i, :, :] = diffs

        # u_ = self.u.detach().flatten()[self.v2d].reshape(-1, 3)
        # difference = u_.unsqueeze_(0).repeat(10, 1, 1) - img
        
        # effs_and_gradients = []
        # effs_and_gradients.append(difference)
    
        # difference.mean().backward()
        # # logging.info(f"matlab_ u_ is {u_.size()}")
        # # effs_and_gradients.append(u_ - img)
        # logging.info(f"matlab_ effs_and_gradients[0] : and {effs_and_gradients[0].size()}")

        # try:
        #     #TODO: increased parameters to be supported
        # effs_and_gradients.append(difference)
        # effs_and_gradients.append(self.mu.grad.detach().numpy()) # since we have to revert it back to tensor
        # effs_and_gradients.append(self.beta.grad.detach().numpy()) # since we have to revert it back to tensor
        # effs_and_gradients.append(self.force.grad.detach().numpy()) # since we have to revert it back to tensor
        # except:
        #     import sys
        #     e = sys.exc_info()[0]
        #     print( "<p>Error: %s</p>" % e )
        
        return difference
        # return self.u.flatten()[self.v2d].reshape(-1, 3).unsqueeze_(0).repeat(10, 1, 1)
