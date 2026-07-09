
import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, layers=(2, 64, 64, 64, 1)):
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

    def forward(self, x, t):
        """
        x, t : tensors of shape (N, 1)
        returns c(x, t) of shape (N, 1)
        """
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)
    
    def save_model(self, path):
        print("\nSaving model...")
        torch.save(self.state_dict(), path)