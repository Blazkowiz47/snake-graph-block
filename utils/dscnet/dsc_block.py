import math
import torch
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential
from utils.gcn_lib.torch_nn import act_layer
from utils.dscnet.Code.DRIVE.S3_DSConv_pro import DSConv_pro


class DSCModule(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 3,
        stride=1,
        remove_dsc: bool = False,
        bias=True,
    ) -> None:
        super(DSCModule, self).__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=math.ceil(kernel / 2) - 1,
            bias=bias,
        )
        if not remove_dsc:
            self.xdsc = DSConv_pro(
                in_channels, out_channels, kernel_size=kernel * kernel, morph=0
            )
            self.ydsc = DSConv_pro(
                in_channels, out_channels, kernel_size=kernel * kernel, morph=1
            )

        self.enc = Conv2d(
            out_channels if remove_dsc else out_channels * 3,
            out_channels,
            kernel,
            stride=stride,
            padding=math.ceil(kernel / 2) - 1,
            bias=bias,
        )
        self.remove_dsc = remove_dsc

    def forward(self, inputs):
        c = self.conv(inputs)
        if not self.remove_dsc:
            x = self.xdsc(inputs)
            y = self.ydsc(inputs)
            return self.enc(torch.cat([c, x, y], dim=1))
        return self.enc(c)


class DSCStem(Module):
    """
    Stem.
    """

    def __init__(
        self,
        in_dim=1,
        total_layers: int = 2,
        out_dim=256,
        act="relu",
        f: int = 3,
        remove_dsc: bool = False,
        bias=True,
        requires_grad=True,
    ):
        super(DSCStem, self).__init__()
        self.layers = []
        start_channels = 16
        for layer_number in range(total_layers):
            self.layers.append(
                DSCModule(
                    in_dim,
                    start_channels,
                    f,
                    stride=2 if layer_number < 2 else 1,
                    bias=bias,
                    remove_dsc=remove_dsc,
                )
            )
            self.layers.append(BatchNorm2d(start_channels))
            if layer_number < 2:
                self.layers.append(act_layer(act))
                in_dim = start_channels
                start_channels = start_channels * 2
            else:
                in_dim = start_channels

        self.stem = Sequential(*self.layers)
        for parameter in self.stem.parameters():
            parameter.requires_grad = requires_grad

    def forward(self, inputs):
        """
        Forward pass.
        """
        inputs = self.stem(inputs)
        return inputs
