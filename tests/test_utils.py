import os

from src.utils.utils import *
from src.core.model_structure import UNet

import numpy as np
from PIL import Image

import pytest
import unittest
import torch

def test_check_cuda_availability():
    device =check_cuda_availability()
    assert device == 'cpu'
    assert type(device) == str

class TestLoadModel(unittest.TestCase):
    def setUp(self):
        self.path = './../../trained_model.pth'
        self.absolute_path = os.path.abspath(self.path)

        self.fake_path = './../../trained_model_fake.pth'
        self.absolute_path_fake = os.path.abspath(self.fake_path)
    def test_load_model(self):
        model = load_model(self.absolute_path)
        assert isinstance (model, UNet)

    def test_load_model_not_found(self):
        model = load_model(self.absolute_path_fake)
        assert model is None

class TestPredict(unittest.TestCase):
    def setUp(self) -> None:
        self.model = UNet()

    def test_predict_valid_input(self):
        fake_image = np.random.rand(256, 256)
        pred_image = predict(self.model, fake_image)
        assert isinstance(pred_image, Image.Image)
        
    def test_fake_predict_valid_input(self):
        fake_image = np.random.rand(256, 257)
        with pytest.raises(Exception):
            pred_image = predict(self.model, fake_image)
