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
        self.sampling_mode = params.generate_samples_mode
        # self.beta = torch.randn(1, 1, requires_grad = True, dtype = torch.float64)
        self.mu_ = params.mu
        self.beta_ = params.beta
        self.force_ = params.force
        self.batch_size_ = params.batch_size_start
        self.mu = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., params.mu+torch.rand(1)[0]*10).requires_grad_(True)
        self.beta = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., params.mu+torch.rand(1)[0]*10).requires_grad_(True)
        self.force = torch.DoubleTensor([[params.force]] * params.batch_size_start)#, ruquires_grad = True)
        self.params = [self.mu, self.beta]
    
    def parameters(self):
        return self.params

    def generate(self):
        if self.sampling_mode:
            self.mu = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.mu_+torch.rand(1)[0]*10).requires_grad_(True)
            self.beta = torch.DoubleTensor(self.batch_size_, 1).uniform_(0., self.beta_+torch.rand(1)[0]*10).requires_grad_(True)
            self.force = torch.DoubleTensor([[self.force_]] * self.batch_size_)#, ruquires_grad = True)
        return [self.mu, self.beta, self.force]