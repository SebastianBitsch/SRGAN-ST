# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import torch
import functools

from torch import Tensor
from torch import nn
import torch.nn.functional as F

from config import Config



def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


# class Generator(nn.Module):
#     def __init__(self, in_nc, out_nc, nf, nb, gc=32):
#         super(Generator, self).__init__()
#         RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

#         self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
#         self.RRDB_trunk = make_layer(RRDB_block_f, nb)
#         self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         #### upsampling
#         self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         fea = self.conv_first(x)
#         trunk = self.trunk_conv(self.RRDB_trunk(fea))
#         fea = fea + trunk

#         fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
#         out = self.conv_last(self.lrelu(self.HRconv(fea)))

#         return out

class Generator(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Generator, self).__init__()

        # Low frequency information extraction layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(config.MODEL.G_IN_CHANNEL, config.MODEL.G_N_CHANNEL, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )

        # High frequency information extraction block
        trunk = []
        for _ in range(config.MODEL.G_N_RCB):
            trunk.append(_ResidualConvBlock(config.MODEL.G_N_CHANNEL))
        self.trunk = nn.Sequential(*trunk)

        # High-frequency information linear fusion layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(config.MODEL.G_N_CHANNEL, config.MODEL.G_N_CHANNEL, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(config.MODEL.G_N_CHANNEL),
        )

        # zoom block
        upsampling = []
        if config.DATA.UPSCALE_FACTOR == 2 or config.DATA.UPSCALE_FACTOR == 4 or config.DATA.UPSCALE_FACTOR == 8:
            for _ in range(int(math.log(config.DATA.UPSCALE_FACTOR, 2))):
                upsampling.append(_UpsampleBlock(config.MODEL.G_N_CHANNEL, 2))
        elif config.DATA.UPSCALE_FACTOR == 3:
            upsampling.append(_UpsampleBlock(config.MODEL.G_N_CHANNEL, 3))
        self.upsampling = nn.Sequential(*upsampling)

        # reconstruction block
        self.conv3 = nn.Conv2d(config.MODEL.G_N_CHANNEL, config.MODEL.G_OUT_CHANNEL, (9, 9), (1, 1), (4, 4))

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 192 x 192
            nn.Conv2d(3, 3, 3, 1, 1, bias=True),
            nn.Conv2d(3, 3, 4, 2, 1, bias=False),
            nn.BatchNorm2d(3, affine=True),
            # input size. (3) x 96 x 96
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

# class Discriminator(nn.Module):
#     def __init__(self, in_chl, nf):
#         super(Discriminator, self).__init__()
#         # in: [in_chl, 192, 192]
#         self.conv0_0 = nn.Conv2d(in_chl, nf, 3, 1, 1, bias=True)
#         self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
#         self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
#         # [nf, 96, 96]
#         self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
#         self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
#         self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
#         self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
#         # [nf*2, 48, 48]
#         self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
#         self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
#         self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
#         self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
#         # [nf*4, 24, 24]
#         self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
#         self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
#         self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
#         self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
#         # [nf*8, 12, 12]
#         self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
#         self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
#         self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
#         self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)
#         # [nf*8, 6, 6]
#         self.conv5_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
#         self.bn5_0 = nn.BatchNorm2d(nf * 8, affine=True)
#         self.conv5_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
#         self.bn5_1 = nn.BatchNorm2d(nf * 8, affine=True)
#         # [nf*8, 3, 3]
#         self.linear1 = nn.Linear(nf * 8 * 3 * 3, 100)
#         self.linear2 = nn.Linear(100, 1)

#         # activation function
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

#     def forward(self, x):
#         fea = self.lrelu(self.conv0_0(x))
#         fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

#         fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
#         fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

#         fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
#         fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

#         fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
#         fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

#         fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
#         fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

#         fea = self.lrelu(self.bn5_0(self.conv5_0(fea)))
#         fea = self.lrelu(self.bn5_1(self.conv5_1(fea)))

#         fea = fea.view(fea.size(0), -1)
#         fea = self.lrelu(self.linear1(fea))
#         out = self.linear2(fea)
#         return out


class _ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.rcb(x)

        out = torch.add(out, identity)

        return out


class _UpsampleBlock(nn.Module):
    def __init__(self, channels: int, upscale_factor: int) -> None:
        super(_UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upscale_factor * upscale_factor, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.upsample_block(x)

        return out
