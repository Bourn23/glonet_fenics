import torch

class engine:
    def __init__(self, fenics_model, wavelength, angle):
        self.model = fenics_model
        self.wavelength = torch.tensor([[wavelength]], requires_grad = True, dtype = torch.float64)
        self.desired_angle = torch.tensor([[angle]], requires_grad = True, dtype = torch.float64)
        self.u = None
    def Eval_Eff_1D_parallel(self, img, wavelength, desired_angle):
        # this only works a single image
        self.u = model(wavelength, desired_angle)
        return self.u - img
    
    def GradientFromSolver_1D_parallel(self, img):
        if self.u is None:
            if type(self.wavelength) != torch.Tensor:
                self.wavelength = torch.tensor([[self.wavelength]], dtype = torch.float64, requires_grad=True)
                self.desired_angle = torch.tensor([[self.desired_angl]], dtype = torch.float64, requires_grad=True)
            print(self.wavelength.size(), self.desired_angle.size())
            self.u = self.model(self.wavelength, self.desired_angle)
        self.u.sum().backward()
        effs_and_gradients = []
        # effs_and_gradients.append(self.u[:, :100, 1] - img[:, :, torch.randint(0, 255, (1,))[0].numpy().tolist()]) # dims needs to match
        effs_and_gradients.append(self.u.sum() - img.sum())
        for param in self.args:
            try:
                effs_and_gradients.append(param.grad)
            except error as e:
                print(f"Error computing gradient {e}")
        
        return effs_and_gradients
