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

    def test_initialization(self):
        """
        Test initialization of the convolution block with correct parameters
        """

        self.assertEqual(self.conv_block.conv1.in_channels, 1)
        self.assertEqual(self.conv_block.conv1.out_channels, 64)
        self.assertEqual(self.conv_block.conv2.out_channels, 64)

    def test_initialization_parameters(self):
        """
        Test the configuration of convolution layers in the Conv_block
        """

        self.assertEqual(self.conv_block.conv1.kernel_size, (3, 3))
        self.assertEqual(self.conv_block.conv2.kernel_size, (3, 3))

        self.assertEqual(self.conv_block.conv1.padding, (1, 1))
        self.assertEqual(self.conv_block.conv2.padding, (1, 1))

    def test_forward_pass(self):
        """
        Test the forward method with random input data
        """

        input_data = torch.rand(1, 1, 256, 256)
        output = self.conv_block(input_data)
        assert output.shape == (1, 64, 256, 256)

    def test_forward_operations(self):
        """
        Manual test the correctness of operations in the forward pass
        """

        import torch.nn as nn
        from torch.nn.functional import relu

        in_channels = 1
        out_channels = 64
        input_data = torch.rand(1, 1, 128, 128)
        conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        output = self.conv_block(input_data)
        manual_output = relu(conv2(relu(conv1(input_data))))

        self.assertEqual(output.shape, manual_output.shape)

    def test_forward_pass_error(self):
        """
        Test if RuntimeError is raised for incorrect input
        """

        input_data = torch.rand(1, 2, 256, 256)

        with self.assertRaises(RuntimeError):
            output = self.conv_block(input_data)
