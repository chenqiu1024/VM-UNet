import torch
import torch.nn as nn

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=False):
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=False):
        super(DecoderBlock, self).__init__(
        )
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv = DoubleConv(2 * out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def center_crop(t, target_shape):
        _, _, th, tw = target_shape
        _, _, h, w = t.shape
        dh = (h - th) // 2
        dw = (w - tw) // 2
        return t[:, :, dh:dh + th, dw:dw + tw]

    def forward(self, x, skip):
        x = self.up(x)
        cropped = self.center_crop(skip, x.shape)
        x = torch.cat([x, cropped], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64, padding=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=padding)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=padding)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=padding)
        self.conv4 = nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=padding)
        self.conv5 = nn.Conv2d(base_channels * 8, base_channels * 16, kernel_size=3, padding=padding)

        self.pool = nn.MaxPool2d(2)
    
        self.decoder1 = DecoderBlock(base_channels * 16, base_channels * 8, kernel_size=3, padding=padding)
        self.decoder2 = DecoderBlock(base_channels * 8, base_channels * 4, kernel_size=3, padding=padding)
        self.decoder3 = DecoderBlock(base_channels * 4, base_channels * 2, kernel_size=3, padding=padding)
        self.decoder4 = DecoderBlock(base_channels * 2, base_channels, kernel_size=3, padding=padding)

        self.head = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv1(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))
        e5 = self.conv5(self.pool(e4))
        d1 = self.decoder1(e5, e4)
        d2 = self.decoder2(d1, e3)
        d3 = self.decoder3(d2, e2)
        d4 = self.decoder4(d3, e1)
        y = self.head(d4)
        ## For debugging only: Check shapes of all the intermediate tensors
        print(f"e1 shape: {e1.shape}")
        print(f"e2 shape: {e2.shape}")
        print(f"e3 shape: {e3.shape}")
        print(f"e4 shape: {e4.shape}")
        print(f"e5 shape: {e5.shape}")
        print(f"d1 shape: {d1.shape}")
        print(f"d2 shape: {d2.shape}")
        print(f"d3 shape: {d3.shape}")
        print(f"d4 shape: {d4.shape}")
        return y