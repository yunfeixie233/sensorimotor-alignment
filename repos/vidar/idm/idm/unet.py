import torch
from torch import nn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = None
        self.conv = doubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        if self.up is None:
            self.up = nn.Upsample(size=x2.size()[2:], mode='bilinear', align_corners=True)
        # 上采样
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        # 经历双卷积
        x = self.conv(x)
        return x


def doubleConv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def down(in_channels,out_channels):
    layer = []
    layer.append(nn.MaxPool2d(2,stride=2))
    layer.append(doubleConv(in_channels, out_channels))
    return nn.Sequential(*layer)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channel=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = 5
        self.in_conv = doubleConv(self.in_channels, base_channel)
        self.downs = [None] * self.num_layers
        self.ups = [None] * self.num_layers
        for i in range(self.num_layers):
            down_in_channel = base_channel * 2 ** i
            down_out_channel = down_in_channel * 2 if i < self.num_layers - 1 else down_in_channel
            up_in_channel = base_channel * 2 ** (self.num_layers - i)
            up_out_channel = up_in_channel // 4 if i < self.num_layers - 1 else base_channel
            self.downs[i] = down(down_in_channel, down_out_channel)
            self.ups[i] = Up(up_in_channel, up_out_channel)
        self.downs = nn.Sequential(*self.downs)
        self.ups = nn.Sequential(*self.ups)
        self.out = nn.Conv2d(in_channels=base_channel, out_channels=self.out_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_conv(x)
        xs = [x]
        for down in self.downs:
            x = down(xs[-1])
            xs.append(x)

        x_out = None
        for x, up in zip(xs[::-1][1:], self.ups):
            if x_out is None:
                x_out = up(xs[-1], xs[-2])
            else:
                x_out = up(x_out, x)
        out = self.out(x_out)
        return out
