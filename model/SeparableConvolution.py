import torch.nn as nn


class SeparableConv2d(nn.Module):
    """
    Xception structure

    The inference speed of SeparableConv2d is faster than nn.Conv2d under fp16.
                        kernel      type        mean        std
    SeparableConv2d      3x3        fp16        293 us      1.39 us
                         5x5        fp16        507 us      0.977 us
    nn.Conv2d            3x3        fp16        435 us      0.775 us
                         5x5        fp16        1.27 ms     6.68 us

    Thanks to the architecture's 3x3 convolution core optimization, nn.Conv2d is only 48.46% slower than SeparableConv2d.
    Under the condition that the convolution kernel is 5x5, nn.Conv2d is 150.49% slower than SeparableConv2d.
    """

    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, dilation=1, bias=False
    ) -> None:
        super().__init__()

        self.pointWiseConv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.depthWiseConv = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation,
                                       groups=out_channels, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.pointWiseConv(x)
        out = self.relu(out)
        out = self.depthWiseConv(out)
        return out


def getSeparableConv2d_3x3(in_channels, out_channels, stride: int = 1, bias=False):
    return SeparableConv2d(in_channels, out_channels, 3, stride, 1, 1, bias=bias)


def getSeparableConv2d_5x5(in_channels, out_channels, stride: int = 2, bias=False):
    return SeparableConv2d(in_channels, out_channels, 5, stride, 2, 1, bias=bias)
