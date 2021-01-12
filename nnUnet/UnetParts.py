""" Parts of the U-Net model """
import torch
import torch.nn as nn
from BuildingBlocks import DoubleConv
from torch.nn import functional as F
from functools import partial


class Encoder(nn.Module):
    """
      Args:
          :param in_channels:(int) number of input channels
          :param out_channels: (int) number of output segmentation masks;
          :param basic_module: (nn.Module) the base module for the net (default is DoubleConv)
          :param apply_pooling: (bool) if true Dimension reduction is done with Max Pool,
          if False it is done with stride=2 in the first conv layer
      """
    def __init__(self, in_channels, out_channels, basic_module=DoubleConv, apply_pooling=False):
        super().__init__()
        if apply_pooling:
            self.basic_module = basic_module(in_channels, out_channels, stride_first_layer=1)
            self.pool = nn.MaxPool3d(kernel_size=2)
        else:
            self.pool = None
            self.basic_module = basic_module(in_channels, out_channels, stride_first_layer=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.basic_module(x)


class Decoder(nn.Module):
    """
      Args:
          :param conv_trans_channels: (int) number of input (and output) channel for the ConvTranspose3d
          :param in_channels:(int) number of input channels
          :param out_channels: (int) number of output segmentation masks;
          :param basic_module: (nn.Module) the base module for the net (default is DoubleConv)
      """
    def __init__(self, conv_trans_channels, in_channels, out_channels, basic_module=DoubleConv, interpolate=False):
        super().__init__()
        if interpolate:
            self.Upsample = partial(self._interpolate, mode='nearest')
        else:
            # Dout=(Din−1)×stride[0]−2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
            self.Upsample = nn.ConvTranspose3d(conv_trans_channels, conv_trans_channels, kernel_size=3,
                                            stride=2, padding=1, output_padding=1)
        self.basic_module = basic_module(in_channels, out_channels, stride_first_layer=1)

    def forward(self, encoder_features, x1):
        output_size = encoder_features.size()[2:]
        x1 = self.Upsample(x1, output_size)
        x = torch.cat([encoder_features, x1], dim=1)
        return self.basic_module(x)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)
