import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np
from PIL import Image


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs):
        x = relu(self.conv1(inputs))
        x = relu(self.conv2(x))

        return x


class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convolution = Conv_block(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = self.convolution(inputs)
        p = self.pool(x)

        return x, p


class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        self.dconv1 = nn.Conv2d(
            in_channels + out_channels, out_channels, kernel_size=3, padding=1
        )
        self.dconv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs, skip):
        x = self.upconv(inputs)
        x = torch.cat([x, skip], dim=1)
        x = relu(self.dconv1(x))
        x = relu(self.dconv2(x))

        return x


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
