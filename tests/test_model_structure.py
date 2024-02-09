import torch
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image

from src.core.model_structure import UNet


class TestUNet(unittest.TestCase):

    """
    Tests Unet model structure
    """

    def setUp(self):
        self.model = UNet()

    def test_initialization(self):
        '''
        Test initialization of the convolution block with correct parameters
        '''

        #Encoder part
        self.assertEqual(self.model.econv1.convolution.conv1.in_channels, 1)
        self.assertEqual(self.model.econv1.convolution.conv1.out_channels, 64)
        self.assertEqual(self.model.econv1.convolution.conv2.out_channels, 64)

        self.assertEqual(self.model.econv2.convolution.conv1.in_channels, 64)
        self.assertEqual(self.model.econv2.convolution.conv1.out_channels, 128)
        self.assertEqual(self.model.econv2.convolution.conv2.out_channels, 128)

        self.assertEqual(self.model.econv3.convolution.conv1.in_channels, 128)
        self.assertEqual(self.model.econv3.convolution.conv1.out_channels, 256)
        self.assertEqual(self.model.econv3.convolution.conv2.out_channels, 256)

        self.assertEqual(self.model.econv4.convolution.conv1.in_channels, 256)
        self.assertEqual(self.model.econv4.convolution.conv1.out_channels, 512)
        self.assertEqual(self.model.econv4.convolution.conv2.out_channels, 512)

        #Bootleneck
        self.assertEqual(self.model.b.conv1.in_channels, 512)
        self.assertEqual(self.model.b.conv1.out_channels, 1024)
        self.assertEqual(self.model.b.conv2.out_channels, 1024)

        #Decoder part
        self.assertEqual(self.model.dconv1.upconv.in_channels, 1024)
        self.assertEqual(self.model.dconv1.dconv1.in_channels, 1024+512)
        self.assertEqual(self.model.dconv1.dconv1.out_channels, 512)
        self.assertEqual(self.model.dconv1.dconv2.out_channels, 512)

        self.assertEqual(self.model.dconv2.upconv.in_channels, 512)
        self.assertEqual(self.model.dconv2.dconv1.in_channels, 512+256)
        self.assertEqual(self.model.dconv2.dconv1.out_channels, 256)
        self.assertEqual(self.model.dconv2.dconv2.out_channels, 256)

        self.assertEqual(self.model.dconv3.upconv.in_channels, 256)
        self.assertEqual(self.model.dconv3.dconv1.in_channels, 256+128)
        self.assertEqual(self.model.dconv3.dconv1.out_channels, 128)
        self.assertEqual(self.model.dconv3.dconv2.out_channels, 128)

        self.assertEqual(self.model.dconv4.upconv.in_channels, 128)
        self.assertEqual(self.model.dconv4.dconv1.in_channels, 128+64)
        self.assertEqual(self.model.dconv4.dconv1.out_channels, 64)
        self.assertEqual(self.model.dconv4.dconv2.out_channels, 64)

        #Output layer
        self.assertEqual(self.model.outconv.in_channels, 64)
        self.assertEqual(self.model.outconv.out_channels, 2)
        self.assertEqual(self.model.outconv.kernel_size, (1, 1))

    def test_forward(self):
        """
        Check that the forward pass returns the expected output shape
        """

        input_data = torch.randn(1, 1, 256, 256)
        output = self.model(input_data)
        self.assertEqual(output.shape, torch.Size([1, 2, 256, 256]))

    @patch.object(UNet, "forward")
    def test_predict(self, mock_forward):
        """
        Mock the forward pass and verify that the prediction returns an Image.Image object
        """

        mock_forward.return_value = torch.ones(1, 2, 256, 256)
        input_data = torch.rand(1, 1, 256, 256)
        result = self.model.predict(input_data)
        self.assertIsInstance(result, Image.Image)

    def test_predict_shape(self):
        """
        Verify that the output prediction shape is as expected
        """

        input_data = torch.rand(1, 1, 256, 256)
        result = self.model.predict(input_data)
        self.assertEqual(result.size, (256, 256))

    def test_predicted_pixels_range(self):
        """
        Checks that the pixel values in the predicted image are within the expected range
        """

        input_data = torch.rand(1, 1, 256, 256)
        rgb_image = np.array(self.model.predict(input_data))
        min_value, max_value = np.min(rgb_image), np.max(rgb_image)
        assert 0 <= min_value <= 255 and 0 <= max_value <= 255

    def test_image_mode(self):
        """
        Check if the predicted image has the expected "RGB" mode
        """

        input_data = torch.rand(1, 1, 256, 256)
        result = self.model.predict(input_data)
        assert result.mode == "RGB"

    @patch.object(UNet, "forward")
    def test_error(self, mock_forward):
        """
        Mock an forward error and whether it raises a RuntimeError during prediction
        """

        mock_forward.return_value = torch.ones(1, 2, 257, 257)

        with self.assertRaises(RuntimeError):
            input_data = torch.rand(1, 2, 256, 256)
            result = self.model.predict(input_data)

    def test_predict_without_input(self):
        """
        Check if predict reports TypeError when None is given as input
        """

        input_data = None

        with self.assertRaises(TypeError):
            result = self.model.predict(input_data)

    def test_invalid_input_dimensions(self):
        """
        Check if the predict function reports a RuntimeError for input data with invalid dimensions.
        """

        input_data = torch.rand(1, 2, 256, 256)

        with self.assertRaises(RuntimeError):
            result = self.model.predict(input_data)
