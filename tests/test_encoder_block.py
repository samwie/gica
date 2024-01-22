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

    def test_forward_pass(self):
        """
        Check the shapes of the results for the tensors returned in the forward
        """

        input_data = torch.rand(1, 1, 256, 256)
        result_x, result_p = self.encoder_block(input_data)
        self.assertEqual(result_x.shape, (1, 64, 256, 256))
        self.assertEqual(result_p.shape, (1, 64, 128, 128))

    @patch.object(Conv_block, "forward")
    def test_pooling(self, mock_forward):
        """
        Checks if the pooling layer is working properly in the encoder layer.
        """

        input_data = torch.rand(1, 1, 256, 256)
        mock_forward.return_value = torch.randn(1, 64, 256, 256)
        result_x, result_p = self.encoder_block(input_data)
        self.assertEqual(result_p.shape, (1, 64, 128, 128))

    def test_pooling_parameters(self):
        """
        Checks if the pooling parameters in the encoder layer are set as expected.
        """

        input_data = torch.rand(1, 1, 256, 256)
        _, _ = self.encoder_block(input_data)
        self.assertEqual(self.encoder_block.pool.kernel_size, 2)
        self.assertEqual(self.encoder_block.pool.stride, 2)
