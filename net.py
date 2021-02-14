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
        # self.mu = torch.tensor([[params.wavelength/100]], requires_grad=True, dtype=torch.float64)
        # self.beta = torch.tensor([[params.angle/100]], requires_grad=True, dtype=torch.float64)
        # self.force = torch.tensor([[params.force/100]], requires_grad=True, dtype=torch.float64)
        self.mu = torch.normal(mean=float(params.force), std=1.).type(torch.float64).unsqueeze(1).requires_grad_(True)
        self.beta = torch.normal(mean=float(params.force), std=1.).type(torch.float64).unsqueeze(1).requires_grad_(True)
        self.force = torch.normal(mean=float(params.angle), std=1.).type(torch.float64).unsqueeze(1).requires_grad_(True)
        self.params = [self.mu, self.beta, self.force]
    
    def parameters(self):
        return self.params
# class Generator(nn.Module):
#     def __init__(self, params):
#         super().__init__()

#         self.noise_dim = params.noise_dims

#         self.gkernel = gkern1D(params.gkernlen, params.gkernsig)

#         self.FC = nn.Sequential(
#             nn.Linear(self.noise_dim, 256),
#             nn.LeakyReLU(0.2),
#             nn.Dropout(p=0.2),
#             nn.Linear(256, 32*16, bias=False),
#             nn.BatchNorm1d(32*16),
#             nn.LeakyReLU(0.2),
#         )

#         self.CONV = nn.Sequential(
#             ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
#             nn.BatchNorm1d(16),
#             nn.LeakyReLU(0.2),
#             ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
#             nn.BatchNorm1d(8),
#             nn.LeakyReLU(0.2),
#             ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
#             nn.BatchNorm1d(4),
#             nn.LeakyReLU(0.2),
#             ConvTranspose1d_meta(4, 1, 5),
#             )
        
#         self.FC1 = nn.Linear(256, self.noise_dim)


#     def forward(self, noise, params):
#         net = self.FC(noise)
#         net = net.view(-1, 16, 32)
#         net = self.CONV(net)    
#         net = self.FC1(net) # borna added
#         net = conv1d_meta(net + noise.unsqueeze(1), self.gkernel)
        
#         # net = conv1d_meta(net , self.gkernel)
#         net = torch.tanh(net* params.binary_amp) * 1.05
        
#         # net = torch.sigmoid(net.view(-1, 176, 3)) # borna added
#         net = torch.tanh(net.view(-1, 176, 3)) # borna added

#         return net



