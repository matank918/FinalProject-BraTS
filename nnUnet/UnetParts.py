""" Parts of the U-Net model """
import torch
import torch.nn as nn
from BuildingBlocks import DoubleConv


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
          :param in_channels:(int) number of input channels
          :param out_channels: (int) number of output segmentation masks;
          :param basic_module: (nn.Module) the base module for the net (default is DoubleConv)
      """

    def __init__(self, in_channels, out_channels, basic_module=DoubleConv):
        super().__init__()
        # Dout=(Din−1)×stride[0]−2×padding[0] + dilation[0]×(kernel_size[0]−1) + output_padding[0] + 1
        self.Upsample = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3,
                                           stride=2, padding=1, output_padding=1)
        double_conv_feat = 2*in_channels
        self.basic_module = basic_module(double_conv_feat, out_channels, stride_first_layer=1)

    def forward(self, encoder_features, x1):
        x1 = self.Upsample(x1)
        x = torch.cat([encoder_features, x1], dim=1)
        return self.basic_module(x)

