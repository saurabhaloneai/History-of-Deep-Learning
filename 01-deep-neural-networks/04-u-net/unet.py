import torch
from torch import nn

class DoubleConv(nn.Module):
    """
    This is just an `combination of conv2d + relu
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """ "
    This content conv2d + pooling
    here we also generate the input for the skip connetion
    """

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        pooled = self.pool(down)
        return down, pooled


class UpSample(nn.Module):
    """
    This content convTransposed + double conv
    help in getting precise loc with skip connection input

    """

    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        # matching dimension wiht padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
        )
        x = torch.cat([x2, x1], dim=1)  # skip connections
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down_conv_1 = DownSample(
            in_channels, 64
        )  # startin decreasing feature channels
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_conv_1 = UpSample(1024, 512)  # startin increasing feature channels
        self.up_conv_2 = UpSample(512, 256)
        self.up_conv_3 = UpSample(256, 128)
        self.up_conv_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_conv_1(b, down_4)  # adding the skip connection
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        out = self.out(up_4)
        return out


import torch
import torch.nn as nn
import torch.optim as optim

def test_unet():
    # Define the model
    model = UNet(in_channels=3, out_channels=1)  # Example for RGB input and single channel output

   
    model.eval()

    #  input tensor with batch size 1, 3 channels, and 256x256 dimensions
    input_tensor = torch.randn(1, 3, 256, 256)

    # Perform a forward pass
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Check the output dimensions
    assert output_tensor.shape == (1, 1, 256, 256), f"Expected output shape (1, 1, 256, 256), but got {output_tensor.shape}"

    print("UNet test passed!")

# Run the test
test_unet()
