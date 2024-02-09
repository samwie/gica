import os
import unittest

import torch
import numpy as np
from PIL import Image
import pytest

from src.core.model_structure import UNet
from src.utils.utils import *

from pathlib import Path


def test_check_cuda_availability():
    """
    # Check if the function returns the expected result for CPU

    """

    device = check_cuda_availability()
    assert device == "cpu"
    assert type(device) == str


class TestLoadModel(unittest.TestCase):

    """
    Test loading model
    """

    def setUp(self):
        self.absolute_path = (
            Path(__file__).resolve().parent.parent.parent / "trained_model.pth"
        )
        self.fake_path = "./../../trained_model_fake.pth"
        self.absolute_path_fake = os.path.abspath(self.fake_path)

    def test_load_model(self):
        """
        Checks that the function correctly loads the model from the specified path
        """

        model = load_model(self.absolute_path)
        assert isinstance(model, UNet)

    def test_load_model_not_found(self):
        """
        Checks if the function returns None for a model path that does not exist
        """

        model = load_model(self.absolute_path_fake)
        assert model is None


class TestPredict(unittest.TestCase):

    """
    Test prediction
    """

    def setUp(self) -> None:
        self.model = UNet()

    def test_predict_valid_input(self):
        """
        Verify that the prediction function returns an Image.Image object for the correct input data
        """
        fake_image = np.random.rand(256, 256)
        pred_image = predict(self.model, fake_image)
        assert isinstance(pred_image, Image.Image)

    def test_fake_predict_valid_input(self):
        """
        Check if the prediction function reports an exception for invalid input data.
        """

        fake_image = np.random.rand(256, 257)
        with pytest.raises(Exception):
            pred_image = predict(self.model, fake_image)
