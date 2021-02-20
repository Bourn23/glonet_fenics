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
        self.each_epoch = params.generate_samples_modes
        # self.beta = torch.randn(1, 1, requires_grad = True, dtype = torch.float64)
        self.mu = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., params.mu+torch.rand(1)[0]*10).requires_grad_(True)
        self.beta = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., params.mu+torch.rand(1)[0]*10).requires_grad_(True)
        self.force = torch.DoubleTensor([[params.force]] * params.batch_size_start)#, ruquires_grad = True)
        self.params = [self.mu, self.beta]
    
    def parameters(self):
        return self.params

    def generate(self):
        if self.each_epoch:
            logging.info('generating new shet')
            self.mu = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., params.mu+torch.rand(1)[0]*10).requires_grad_(True)
            self.beta = torch.DoubleTensor(params.batch_size_start, 1).uniform_(0., params.mu+torch.rand(1)[0]*10).requires_grad_(True)
            self.force = torch.DoubleTensor([[params.force]] * params.batch_size_start)#, ruquires_grad = True)
        return [self.mu, self.beta, self.force]