import torch
from torch import nn

from model.SeparableConvolution import getSeparableConv2d_5x5, getSeparableConv2d_3x3


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride: int) -> None:
        super().__init__()

        self.conv1 = getSeparableConv2d_5x5(in_channels, mid_channels, stride)
        self.conv2 = getSeparableConv2d_5x5(mid_channels, out_channels, 1)

        self.proj = nn.Identity() if stride == 1 and in_channels == out_channels else getSeparableConv2d_3x3(
            in_channels, out_channels, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        proj = self.proj(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = x + proj
        return self.relu(x)


def EncoderStage(in_channels, out_channels, num_blocks: int, dropout_p: float):
    blocks = [
        EncoderBlock(
            in_channels=in_channels,
            mid_channels=out_channels // 4,
            out_channels=out_channels,
            stride=2
        )
    ]
    for _ in range(num_blocks - 1):
        blocks.append(
            EncoderBlock(
                in_channels=out_channels,
                mid_channels=out_channels // 4,
                out_channels=out_channels,
                stride=1
            )
        )

    blocks.append(nn.Dropout2d(p=dropout_p))

    return nn.Sequential(*blocks)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.conv0 = getSeparableConv2d_3x3(in_channels, out_channels)
        self.conv1 = getSeparableConv2d_3x3(out_channels, out_channels)

    def forward(self, x):
        inp = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = x + inp
        return x


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_in_channels, out_channels) -> None:
        super().__init__()

        self.decode_conv = DecoderBlock(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.proj_conv = getSeparableConv2d_3x3(skip_in_channels, out_channels)

    def forward(self, inputs):
        inp, skip = inputs

        x = self.decode_conv(inp)
        x = self.upsample(x)
        y = self.proj_conv(skip)
        return x + y


class Network(nn.Module):
    def __init__(self, mc_nums: int = 10, dropout_p: float = 0.2) -> None:
        super().__init__()

        self.mc_nums = mc_nums

        # self.conv0 = getSeparableConv2d_3x3(in_channels=3, out_channels=16)
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )
        self.enc1 = EncoderStage(in_channels=16, out_channels=64, num_blocks=4, dropout_p=dropout_p)
        self.enc2 = EncoderStage(in_channels=64, out_channels=128, num_blocks=4, dropout_p=dropout_p)
        self.enc3 = EncoderStage(in_channels=128, out_channels=256, num_blocks=4, dropout_p=dropout_p)
        self.enc4 = EncoderStage(in_channels=256, out_channels=512, num_blocks=4, dropout_p=dropout_p)

        self.encdec = getSeparableConv2d_3x3(in_channels=512, out_channels=64)
        self.dec1 = DecoderStage(in_channels=64, skip_in_channels=256, out_channels=64)
        self.dec2 = DecoderStage(in_channels=64, skip_in_channels=128, out_channels=32)
        self.dec3 = DecoderStage(in_channels=32, skip_in_channels=64, out_channels=32)
        self.dec4 = DecoderStage(in_channels=32, skip_in_channels=16, out_channels=16)

        self.out0 = DecoderBlock(in_channels=16, out_channels=16)
        # self.out1 = getSeparableConv2d_3x3(16, 3)
        self.out1 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1, stride=1)

    def enableMonteCarloDropout(self):
        for m in self.modules():
            if m.__class__.__name__.startswith('Dropout') or isinstance(m, nn.Dropout2d):
                m.train()

    def forward(self, inp):

        conv0 = self.conv0(inp)
        conv1 = self.enc1(conv0)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)

        conv5 = self.encdec(conv4)

        up3 = self.dec1((conv5, conv3))
        up2 = self.dec2((up3, conv2))
        up1 = self.dec3((up2, conv1))
        up0 = self.dec4((up1, conv0))

        out0 = self.out0(up0)
        out1 = self.out1(out0)

        pred = torch.sigmoid(out1)

        return pred


def test():
    image = torch.randn(1, 3, 256, 256, dtype=torch.float16).cuda()
    net = Network().half().cuda()
    net(image)


if __name__ == '__main__':
    test()
