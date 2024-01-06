import torch.nn as nn
from torch.nn.functional import relu
from torch import Tensor

class Conv_block(nn.Module):
    '''Single convolution block
    '''
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        '''Performs forward pass through the convolution block
        '''
        x = relu(self.conv1(inputs))
        x = relu(self.conv2(x))

        return x