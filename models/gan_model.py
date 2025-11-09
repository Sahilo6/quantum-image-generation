# gan_model.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    """Generator: noise (nz) -> 1x28x28 image"""
    def __init__(self, nz=100):
        super().__init__()
        self.nz = nz
        self.net = nn.Sequential(
            nn.Linear(nz, 128*7*7),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 7->14
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),    # 14->28
            nn.Sigmoid()  # outputs [0,1]
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """Discriminator: 1x28x28 -> logit"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),   # 28->14
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 14->7
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*7*7, 1)        # raw logit
        )

    def forward(self, x):
        return self.net(x).view(-1)
