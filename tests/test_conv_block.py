import unittest
import torch

from src.core.conv_block import Conv_block


class TestConv_block(unittest.TestCase):

    """
    Tests for Convolution block
    """

    def setUp(self):
        """
        Set up the Conv_block instance with input channels 1 and output channels 64
        """

        self.conv_block = Conv_block(1, 64)

    def test_forward_pass(self):
        """
        Test the forward method with random input data
        """import unittest
import torch
from src.core.encoder_block import Encoder_block
from src.core.conv_block import Conv_block
from unittest.mock import MagicMock, patch

        input_data = torch.rand(1, 1, 256, 256)
        output = self.conv_block(input_data)
        assert output.shape == (1, 64, 256, 256)

    def test_conv_layers(self):
        """
        Test the configuration of convolution layers in the Conv_block
        """

        assert self.conv_block.conv1.kernel_size == (3, 3)
        assert self.conv_block.conv1.padding == (1, 1)
        assert self.conv_block.conv2.kernel_size == (3, 3)
        assert self.conv_block.conv2.padding == (1, 1)
