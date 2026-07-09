"""

The memory kernel network K(delta_t): a small independent network that
learns the memory kernel instead of assuming a fixed form like e^{-t}.

"""

import torch
import torch.nn as nn


class KernelNet(nn.Module):
    def __init__(self, layers=(1, 32, 32, 1)):
        super().__init__()
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, dt):
        """
        dt : tensor of shape (N, 1) 
        
        time lag (t - s), always >= 0
        
        returns K(dt) of shape (N, 1)
        """
        return self.net(dt)


class FixedExponentialKernel(nn.Module):
    """
    known kernel exponential
    """

    def __init__(self):
        super().__init__()

    def forward(self, dt):
        return torch.exp(-dt)
