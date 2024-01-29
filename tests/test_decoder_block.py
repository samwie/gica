import unittest
import torch

from src.core.decoder_block import Decoder_block


class TestDecoder_block(unittest.TestCase):

    """
    Tests for Convolution block
    """

    def setUp(self):
        self.decoder_block = Decoder_block(128, 64)

    def test_initialization(self):
        """
        Test initialization of the decoder block with correct parameters
        """

        in_channels = 3
        out_channels = 17
        decoder = Decoder_block(in_channels, out_channels)

        self.assertEqual(decoder.upconv.in_channels, in_channels)
        self.assertEqual(decoder.upconv.out_channels, in_channels)
        self.assertEqual(decoder.dconv1.in_channels, in_channels + out_channels)
        self.assertEqual(decoder.dconv1.out_channels, out_channels)
        self.assertEqual(decoder.dconv2.in_channels, out_channels)
        self.assertEqual(decoder.dconv2.out_channels, out_channels)

    def test_initialization_parameters(self):
        """
        Test the parameters of the upconvolution and convolution operations
        """

        self.assertEqual(self.decoder_block.upconv.kernel_size, (2, 2))
        self.assertEqual(self.decoder_block.upconv.stride, (2, 2))

        self.assertEqual(self.decoder_block.dconv1.kernel_size, (3, 3))
        self.assertEqual(self.decoder_block.dconv1.padding, (1, 1))

        self.assertEqual(self.decoder_block.dconv2.kernel_size, (3, 3))
        self.assertEqual(self.decoder_block.dconv2.padding, (1, 1))

    def test_foward_pass(self):
        """
        Test the forward method with random input data
        """

        input_data = torch.randn(1, 128, 128, 128)
        input_skip = torch.rand(1, 64, 256, 256)
        output = self.decoder_block(input_data, input_skip)
        self.assertEqual(output.shape, (1, 64, 256, 256))

    def test_forward_operations(self):
        """
        Manual test the correctness of operations in the forward pass
        """

        import torch.nn as nn
        from torch.nn.functional import relu
        import torch.testing

        in_channels = 128
        out_channels = 64
        input_data = torch.rand(1, 128, 128, 128)
        input_skip = torch.rand(1, 64, 256, 256)

        output = self.decoder_block(input_data, input_skip)

        upconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        dconv1 = nn.Conv2d(
            in_channels + out_channels, out_channels, kernel_size=3, padding=1
        )
        dconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        manual_output = relu(
            dconv2(relu(dconv1(torch.cat([upconv(input_data), input_skip], dim=1))))
        )
        self.assertEqual(output.shape, manual_output.shape)

    def test_foward_RuntimeError(self):
        """
        Test if RuntimeError is raised for incorrect input
        """

        input_data = torch.randn(1, 128, 128, 128)
        input_skip = torch.rand(1, 128, 256, 256)
        with self.assertRaises(RuntimeError):
            self.decoder_block(input_data, input_skip)
