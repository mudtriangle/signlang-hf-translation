import torch
from torch import nn


class LinearAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.l = nn.Linear(in_dim, out_dim)
        self.name = name

    def forward(self, x):
        x = x.type(torch.float32)
        x = self.l(x)
        return x

class ConvAdapter(nn.Module):
    def __init__(self, in_channels, out_channels, n_patches_height, n_patches_width):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (1, n_patches_height, n_patches_width))

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv(x)
        x = torch.squeeze(x, dim=(3, 4))
        return x
