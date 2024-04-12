from dataclasses import dataclass
import math
from typing import Optional
import torch
from torch.nn import (
    AvgPool2d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)

from models.config import Config
from models.predictor import Predictor


class EcaModule(Module):
    """Constructs an ECA module.

    Args:
        channels: Number of channels of the input feature map for use in adaptive kernel sizes
            for actual calculations according to channel.
            gamma, beta: when channel is given parameters of mapping function
            refer to original paper https://arxiv.org/pdf/1910.03151.pdf
            (default=None. if channel size not given, use k_size given for kernel size.)
        kernel_size: Adaptive selection of kernel size (default=3)
    """

    def __init__(self, channels=None, kernel_size=3, gamma=2, beta=1):
        super(EcaModule, self).__init__()
        assert kernel_size % 2 == 1
        if channels is not None:
            t = int(abs(math.log(channels, 2) + beta) / gamma)
            kernel_size = max(t if t % 2 else t + 1, 3)

        self.conv = Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )

    def forward(self, x):
        y = x.mean((2, 3)).view(x.shape[0], 1, -1)  # view for 1d conv
        y = self.conv(y)
        y = y.view(x.shape[0], -1, 1, 1).sigmoid()
        return x * y.expand_as(x)


class ECAResnetBlock(Module):
    def __init__(self, channels) -> None:
        super(ECAResnetBlock, self).__init__()
        self.layers = [
            Conv2d(channels, channels, 3, 1, 1),
            BatchNorm2d(channels),
            ReLU(),
            Conv2d(channels, channels, 3, 1, 1),
            BatchNorm2d(channels),
        ]
        self.resnet_block = Sequential(*self.layers)
        self.eca = EcaModule(channels)
        self.act = ReLU()

    def forward(self, input):
        x = input
        #        print("Input: ", x.shape)
        x = self.resnet_block(x)
        #        print("After Resnet: ", x.shape)
        y = self.eca(x)
        x = x + y
        x = self.act(x)
        return x


class Model(Module):
    """
    Arc Vein Model.
    """

    def __init__(self) -> None:
        super(Model, self).__init__()
        img_dim, channels = 224, 3
        self.convblock1 = Sequential(
            Conv2d(channels, 64, 7, 2, 5),
            MaxPool2d(3, 2),
        )
        img_dim = img_dim // 2
        channels = 64
        self.convblock2 = ECAResnetBlock(channels)
        img_dim = img_dim // 2
        self.c1 = Conv2d(channels, channels * 2, 3, 2, 1)
        channels *= 2
        self.convblock3 = ECAResnetBlock(channels)
        img_dim = img_dim // 2
        self.c2 = Conv2d(channels, channels * 2, 3, 2, 1)
        channels *= 2
        self.convblock4 = ECAResnetBlock(channels)
        img_dim = img_dim // 2
        self.c3 = Conv2d(channels, channels * 2, 3, 2, 1)
        channels *= 2
        self.convblock5 = ECAResnetBlock(channels)
        self.avg = AvgPool2d(7)
        self.model_init()

    def forward(self, x):
        """
        Forward pass.
        """
        x = torch.cat([x, x, x], dim=1)
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.c1(x)
        x = self.convblock3(x)
        x = self.c2(x)
        x = self.convblock4(x)
        x = self.c3(x)
        x = self.convblock5(x)
        x = self.avg(x)
        x = torch.squeeze(x)
        return x

    def model_init(self):
        """
        Model init.
        """
        for module in self.modules():
            if isinstance(module, Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                module.weight.requires_grad = True
                if module.bias is not None:
                    module.bias.data.zero_()
                    module.bias.requires_grad = True


