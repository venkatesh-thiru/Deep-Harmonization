import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math
from fft_conv_pytorch import FFTConv2d


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class GeneratorResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
        super(GeneratorResNet, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(1):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 432, 3, 1, 1),
                nn.BatchNorm2d(432),
                nn.PixelShuffle(upscale_factor = 3),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(FFTConv2d(in_channels=48, out_channels=out_channels,kernel_size=27,stride=1, padding=12),nn.PReLU())
        # self.conv3 = nn.Sequential(nn.Conv2d(48, out_channels, kernel_size=27, stride=1, padding=12), nn.PReLU())
        self.conv_resize = nn.Sequential(nn.Conv2d(out_channels,out_channels,1,1,0),nn.PReLU())

    def forward(self, x):
        out1 = self.conv1(x)
        # print(f"first conv : {out1.shape}")
        out = self.res_blocks(out1)
        # print(f"Res Block : {out.shape}")
        out2 = self.conv2(out)
        # print(f"second conv : {out2.shape}")
        out = torch.add(out1, out2)
        # print(f"out : {out.shape}")
        out = self.upsampling(out)
        # print(f"upsample : {out.shape}")
        out = self.conv3(out)
        # print(f"final : {out.shape}")
        # out = self.conv_resize(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)


if __name__ == '__main__':
    import time

    model = GeneratorResNet(in_channels=6, out_channels=6, n_residual_blocks=16).cuda()
    dimensions = 10, 6, 86, 86
    x = torch.randn(dimensions).cuda()
    start_time = time.time()
    x = model(x)
    print(f"EXECUTION TIME : {time.time() - start_time}")
    # print(model)
