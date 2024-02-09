import unittest
from unittest.mock import MagicMock, patch
import torch

from src.core.conv_block import Conv_block
from src.core.encoder_block import Encoder_block


class TestEncoder_block(unittest.TestCase):

    """
    Tests Encoder block
    """

    def setUp(self):
        self.encoder_block = Encoder_block(1, 64)

    def test_initialization(self):
        """
        Test initialization of the encoder block with correct parameters
        """

        self.assertIsInstance(self.encoder_block.convolution, Conv_block)
        self.assertEqual(self.encoder_block.convolution.conv1.in_channels, 1)
        self.assertEqual(self.encoder_block.convolution.conv1.out_channels, 64)
        self.assertEqual(self.encoder_block.convolution.conv2.out_channels, 64)

    def test_initialization_parameters(self):
        """
        Test initialization of the encoder block with correct parameters
        """

        self.assertEqual(self.encoder_block.convolution.conv1.kernel_size, (3, 3))
        self.assertEqual(self.encoder_block.convolution.conv1.padding, (1, 1))
        self.assertEqual(self.encoder_block.convolution.conv2.kernel_size, (3, 3))
        self.assertEqual(self.encoder_block.convolution.conv2.padding, (1, 1))

        self.assertEqual(self.encoder_block.pool.kernel_size, 2)
        self.assertEqual(self.encoder_block.pool.stride, 2)

    def test_forward_pass(self):
        """
        Check the shapes of the results for the tensors returned in the forward
        """

        input_data = torch.rand(1, 1, 256, 256)
        result_x, result_p = self.encoder_block(input_data)
        self.assertEqual(result_x.shape, (1, 64, 256, 256))
        self.assertEqual(result_p.shape, (1, 64, 128, 128))

    def test_forward_operations(self):
        """
        Manual test the correctness of operations in the forward pass
        """

        import torch.nn as nn

        in_channels = 1
        out_channels = 64
        input_data = torch.rand(1, 1, 128, 128)

        convolution = Conv_block(in_channels, out_channels)
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        x, p = self.encoder_block(input_data)
        x_manual = convolution(input_data)
        p_manual = pool(x_manual)

        self.assertEqual(x.shape, x_manual.shape)
        self.assertEqual(p.shape, p_manual.shape)

    @patch.object(Conv_block, "forward")
    def test_pooling(self, mock_forward):
        """
        Checks if the pooling layer is working properly in the encoder layer.
        """

        input_data = torch.rand(1, 1, 256, 256)
        mock_forward.return_value = torch.randn(1, 64, 256, 256)
        result_x, result_p = self.encoder_block(input_data)
        self.assertEqual(result_p.shape, (1, 64, 128, 128))

    def test_forward_invalid_input(self):
        """
        Check the shapes of the results for the tensors returned in the forward
        """

        input_data = torch.rand(1, 2, 256, 256)
        with self.assertRaises(RuntimeError):
            result_x, result_p = self.encoder_block(input_data)
