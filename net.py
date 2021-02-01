import sys
sys.path.append('../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from metalayers import * 

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dims

        self.gkernel = gkern1D(params.gkernlen, params.gkernsig)

        self.FC = nn.Sequential(
            nn.Linear(self.noise_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(256, 32*16, bias=False),
            nn.BatchNorm1d(32*16),
            nn.LeakyReLU(0.2),
        )

        self.CONV = nn.Sequential(
            ConvTranspose1d_meta(16, 16, 5, stride=2, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(16, 8, 5, stride=2, bias=False),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(8, 4, 5, stride=2, bias=False),
            nn.BatchNorm1d(4),
            nn.LeakyReLU(0.2),
            ConvTranspose1d_meta(4, 2, 5),
            )


    def forward(self, noise, params):
        # logging.info(f"noise size: {noise.size()} and type: {type(noise)}")
        net = self.FC(noise)
        logging.info(f"1st pass size: {net.size()}")

        net = net.view(-1, 16, 32)
        logging.info(f"2st pass size: {net.size()}")

        net = self.CONV(net)    
        logging.info(f"3st pass size: {net.size()}")
        logging.info(f"3st pass noise_size: {noise.unsqueeze(1).size()}")
        
        net = conv1d_meta(net + noise.unsqueeze(1), self.gkernel)
        logging.info(f"4st pass size: {net.size()}")
        
        # net = conv1d_meta(net , self.gkernel)
        net = torch.tanh(net* params.binary_amp) * 1.05
        logging.info(f"5st pass size: {net.size()}")

        return net



