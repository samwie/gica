import torch.nn as nn
from .conv_block import Conv_block

class Encoder_block(nn.Module):
    '''Single convolution block
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = Conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        '''Performs forward pass through the encoder block
        '''
        x = self.convolution(inputs)
        p = self.pool(x)

        return x, p