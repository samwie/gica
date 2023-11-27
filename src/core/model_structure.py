import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np
from PIL import Image
from Conv_block import Conv_block
from Decoder_block import Decoder_block
from Encoder_block import Encoder_block

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder part
        self.econv1 = Encoder_block(1, 64)
        self.econv2 = Encoder_block(64, 128)
        self.econv3 = Encoder_block(128, 256)
        self.econv4 = Encoder_block(256, 512)

        # Bootleneck
        self.b = Conv_block(512, 1024)

        # Decoder part
        self.dconv1 = Decoder_block(1024, 512)
        self.dconv2 = Decoder_block(512, 256)
        self.dconv3 = Decoder_block(256, 128)
        self.dconv4 = Decoder_block(128, 64)

        # Output layer
        self.outconv = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, inputs):
        # Encoder part
        x1, p1 = self.econv1(inputs)
        x2, p2 = self.econv2(p1)
        x3, p3 = self.econv3(p2)
        x4, p4 = self.econv4(p3)

        # Bootleneck
        b = self.b(p4)

        # Decoder part
        d1 = self.dconv1(b, x4)
        d2 = self.dconv2(d1, x3)
        d3 = self.dconv3(d2, x2)
        d4 = self.dconv4(d3, x1)

        # Output layer
        out = self.outconv(d4)

        return out

    def predict(self, input_data):
        self.eval()

        with torch.no_grad():
            pred = self.forward(input_data)

        prediction = torch.cat((input_data, pred), dim=1)
        prediction[:, 0, :, :] += 1.0
        prediction[:, 0, :, :] *= 50.0
        prediction[:, 1, :, :] *= 110.0
        prediction[:, 2, :, :] *= 110.0
        pred_arr = prediction.numpy()

        from skimage import color

        rgb_image = np.transpose(pred_arr, (0, 2, 3, 1))
        rgb_image = color.lab2rgb(rgb_image[0])
        rgb_image = np.clip(rgb_image, 0, 1) * 255
        rgb_image = rgb_image.astype(np.uint8)
        image_pil = Image.fromarray(rgb_image)

        self.train()
        return image_pil
