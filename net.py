import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Generator:
    def __init__(self, params):

        # self.beta = torch.randn(1, 1, requires_grad = True, dtype = torch.float64)
        self.mu = torch.DoubleTensor(params.batch_size, 1).uniform_(0., params.mu + torch.rand(1)[0]).requires_grad_(True)
        self.beta = torch.DoubleTensor(params.batch_size, 1).uniform_(0., params.mu + torch.rand(1)[0]).requires_grad_(True)
        self.force = torch.DoubleTensor([[params.force]] * params.batch_size)#, ruquires_grad = True)
        self.params = [self.mu, self.beta]
    
    def parameters(self):
        return self.params

    def generate(self):
        return self.params + [self.force]