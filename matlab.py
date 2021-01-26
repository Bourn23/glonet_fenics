class engine:
    def __init__(self, fenics_model, *args):
        self.model = fenics_model()
        self.args = args
        self.u = None
    def Eval_Eff_1D_parallel(self, img, wavelength, desired_angle):
        # this only works a single image
        self.u = model(wavelength, desired_angle)
        return self.u - img
    
    def GradientFromSolver_1D_parallel(self, img):
        self.u.sum().backward()
        effs_and_gradients = []
        effs_and_gradients.append(self.u - img)
        for param in self.args:
            try:
                effs_and_gradients.append(param.grad)
            except error as e:
                print(f"Error computing gradient {e}")
        
        return effs_and_gradients
