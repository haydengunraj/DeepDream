import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, input_channels, dropout=None, embedding_mode=False, input_size=None, embedding_size=128):
        super().__init__()
        # Contracting side
        self.down1 = _DownBlock(input_channels, 64)
        self.down2 = _DownBlock(64, 128)
        self.down3 = _DownBlock(128, 256)
        self.down4 = _DownBlock(256, 512, dropout=dropout)

        # Bottleneck
        self.bottleneck = _DoubleConvBlock(512, 1024, dropout=dropout)

        # Embedding version
        self.embedding = embedding_mode
        self.embedding_size = embedding_size
        self.embed = _EmbedBlock(1024, self.embedding_size, input_size//16)

        # Expanding side
        self.up1 = _UpBlock(1024, 512)
        self.up2 = _UpBlock(512, 256)
        self.up3 = _UpBlock(256, 128)
        self.up4 = _UpBlock(128, 64)

        # Output layer
        self.out = nn.Conv3d(64, 1, 1, stride=1, padding=0)

    def forward(self, x):
        # Contracting side
        down1, pool1 = self.down1(x)
        down2, pool2 = self.down2(pool1)
        down3, pool3 = self.down3(pool2)
        down4, pool4 = self.down4(pool3)

        # Bottleneck
        bottleneck = self.bottleneck(pool4)

        # Embedding mode
        if self.embedding:
            return self.embed(bottleneck)

        # Expanding side
        up1 = self.up1(bottleneck, down4)
        up2 = self.up2(up1, down3)
        up3 = self.up3(up2, down2)
        up4 = self.up4(up3, down1)

        # Output layer
        out = self.out(up4)

        return out

    def embedding_mode(self, mode=True):
        self.embedding = mode


class _DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=None):
        super().__init__()
        self.double_conv = _DoubleConvBlock(in_channels, out_channels, dropout)
        self.pool = nn.MaxPool3d(2, stride=2)

    def forward(self, x):
        conv = self.double_conv(x)
        pool = self.pool(conv)
        return conv, pool


class _UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels//2, 2, stride=2, padding=0)
        self.double_conv = _DoubleConvBlock(in_channels, out_channels)

    def forward(self, up, down):
        up = self.up(up)
        concat = torch.cat([down, up], dim=1)
        conv = self.double_conv(concat)
        return conv


class _DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1)
        self.dropout = nn.Dropout3d(dropout) if dropout is not None else None

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class _EmbedBlock(nn.Module):
    def __init__(self, in_channels, embedding_size, input_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.conv = nn.Conv3d(in_channels, in_channels//2, 5, stride=1, padding=0)
        linear_in = in_channels//2*(input_size - 4)**3
        self.linear = nn.Linear(linear_in, embedding_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.view((x.size(0), -1))
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x


if __name__ == '__main__':
    import numpy as np

    # 3D network
    x = torch.from_numpy(np.random.random((2, 1, 80, 80, 80))).float()
    net = UNet3D(1, dropout=0.8, embedding_mode=False, input_size=80, embedding_size=128)
    # out = net.forward(x)
    #
    # print(out.size())

    net.embedding_mode()
    out = net.forward(x)

    print(out.size())
    print(torch.norm(out[0], dim=0))  # should be 1


