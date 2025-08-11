import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Conv2d: Performs a 2D convolution between the input and filters (kernels) to extract spatial features. Each filter produces an activation map, and you can specify padding, stride, and other parameters to control the size and behavior of the output.

BatchNorm2d: Normalizes the activations of a convolutional layer across the batch, maintaining a controlled mean and variance. Stabilizes and accelerates training by reducing the shift in the distribution of inputs to subsequent layers.

ReLU (Rectified Linear Unit): Applies the nonlinear function f(x)=max⁡(0,x)f(x)=max(0,x). It clamps negative values to zero, leaving positive values unchanged. It helps introduce nonlinearity and avoid vanishing gradients.

MaxPool2d: Reduces spatial resolution by selecting the maximum value within non-overlapping or partially overlapping windows (kernels). It is useful for summarizing local features and increasing robustness to shifts.

ConvTranspose2d (or "deconvolution"): The inverse of standard convolution. It increases spatial resolution (upsampling) by generating larger outputs from smaller feature maps. It is often used in neural network decoders.

Softmax: Normalizes a vector so that the sum of its components equals 1, applying exponentials and division. It transforms raw values (logits) into probabilities, usually along a specific dimension.
"""



class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, mid_channels=None, padding=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_block(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv_block(in_channels, out_channels, kernel_size),
        )

    def forward(self, x):
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.catconv = Conv_block(out_channels * 2, out_channels, kernel_size, mid_channels=out_channels // 2)

    def forward(self, x, y):
        x = self.up(x)
        x = self.conv(x)
        crop_size = int(y.shape[3] - x.shape[3]) / 2
        _, _, h, w = y.shape
        item_cropped = y[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        return self.catconv(x)


class T_Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, n_filters, n_classes, i, kernel_size=3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)

        self.conv = Conv_block(in_channels, out_channels, kernel_size, mid_channels=out_channels // 2)
        self.catconv = Conv_block(out_channels * 2, out_channels, kernel_size, mid_channels=out_channels // 2)

        ratio = 4
        if i == 'up1':
            self.Pre = nn.Conv2d(out_channels, n_classes, kernel_size=1)
            self.queryd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.keyd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.valued = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

        elif i == 'up2':
            self.Pre = nn.Conv2d(out_channels, n_classes, kernel_size=1)
            self.queryd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.keyd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.valued = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

        elif i == 'up3':
            self.Pre = nn.Conv2d(out_channels, n_classes, kernel_size=1)
            self.queryd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.keyd = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.valued = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.softmaxd = nn.Softmax(dim=-1)
            self.gammad = nn.Parameter(torch.zeros(1))

    def forward(self, x, y, z):
        crop_size = int(z.shape[3] - x.shape[3]) / 2
        _, _, h, w = z.shape
        item_cropped = z[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        B, C, H, W = x.shape
        Q = self.queryd(x).view(B, -1, H * W).permute(0, 2, 1).contiguous()
        K = self.keyd(x).view(B, -1, H * W)
        _, head_dim, _ = K.shape
        attention = self.softmaxd(torch.bmm(Q, K) / (head_dim ** 0.5))
        V = self.valued(x).view(B, C, H * W).permute(0, 2, 1).contiguous()
        out = torch.bmm(attention, V).permute(0, 2, 1).contiguous().view(B, C, H, W)
        x = self.gammad * out + x
        x = self.up(x)
        x = self.conv(x)

        crop_size = int(y.shape[3] - x.shape[3]) / 2
        _, _, h, w = y.shape
        item_cropped = y[:, :, int(crop_size):h - int(crop_size), int(crop_size):w - int(crop_size)]
        x = torch.cat((x, item_cropped), dim=1)
        output = self.catconv(x)
        return output, self.Pre(output)


class Output(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        )

    def forward(self, x):
        return self.output(x)
