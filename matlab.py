import torch
import logging

class engine:
    def __init__(self, fenics_model, wavelength, angle, batch_size):
        self.model = fenics_model
        self.wavelength = torch.tensor([[wavelength]] * params.batch_size, requires_grad = True, dtype = torch.float64)
        self.desired_angle = torch.tensor([[angle]] * params.batch_size, requires_grad = True, dtype = torch.float64)
        self.u = None
        # logging.info(f"matlab_All initialized: type of wave {type(self.wavelength)} and angle {type(self.desired_angle)}")
    def Eval_Eff_1D_parallel(self, img, wavelength, desired_angle):
        logging.info("matlab_eval_eff_called")
        # this only works a single image
        if self.u is None:
            if type(self.wavelength) != torch.Tensor:
                logging.info("matlab_EVAL_EFF_data_U_Unknown")
                self.wavelength = torch.tensor([[self.wavelength]] * params.batch_size, dtype = torch.float64, requires_grad=True)
                self.desired_angle = torch.tensor([[self.desired_angl]] * params.batch_size, dtype = torch.float64, requires_grad=True)
            self.u = self.model(self.wavelength, self.desired_angle)

        return (self.u.sum() - img.sum()).float()
    
    def GradientFromSolver_1D_parallel(self, img):
        # What should be going on here? 1. see paper for what they're doing / 2. see the .mat file
        # And what is going on now?:))
        logging.info("matlab_grad_solver_called")
        if self.u is None:
            if type(self.wavelength) != torch.Tensor:
                logging.info("matlab_EVAL_EFF_data_U_Unknown_TENSORS")
                self.wavelength = torch.tensor([[self.wavelength]] * params.batch_size, dtype = torch.float64, requires_grad=True)
                self.desired_angle = torch.tensor([[self.desired_angl]] * params.batch_size, dtype = torch.float64, requires_grad=True)

            # compute gradients for all!
            self.u = self.model(self.wavelength, self.desired_angle)
        logging.info(f"self.u.size before is : {self.u.size()}")
        self.u.mean().backward(retain_graph = True) # mean(axis = 0) to average over batches I'm thinking how to calculate gradients for each and one of them
        # self.u.sum().backward(retain_graph = True)

        logging.info(f"self.u.size is : {self.u.size()}")
        logging.info(f"img size is : {img.size()}")

        effs_and_gradients = []
        # effs_and_gradients.append(self.u[:, :100, 1] - img[:, :, torch.randint(0, 255, (1,))[0].numpy().tolist()]) # dims needs to match
        effs_and_gradients.append(self.u - img)

        try:
            effs_and_gradients.append(self.wavelength.grad)
            effs_and_gradients.append(self.desired_angle.grad)
        except:
            import sys
            e = sys.exc_info()[0]
            print( "<p>Error: %s</p>" % e )
        
        return effs_and_gradients
