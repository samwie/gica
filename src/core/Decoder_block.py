import torch
import torch.nn as nn
from torch.nn.functional import relu

class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.dconv1 = nn.Conv2d(
            in_channels + out_channels, out_channels, kernel_size=3, padding=1
        )
        self.dconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs, skip):
        x = self.upconv(inputs)
        x = torch.cat([x, skip], dim=1)
        x = relu(self.dconv1(x))
        x = relu(self.dconv2(x))

        return x