import torch.nn as nn
from utils import config as cfg


class DoubleConv(nn.Module):
    """(convolution => [IN] => LeakyReLU) * 2
      Args:
          :param in_channels:(int) number of input channels
          :param out_channels: (int) number of output segmentation masks;
          :param stride_first_layer: (int) stride for the first conv layer
      """

    def __init__(self, in_channels, out_channels, stride_first_layer):
        super().__init__()
        # D_out=((D_in+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/stride[0])+1
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=cfg.kernel_size, padding=cfg.padding,
                      stride=stride_first_layer),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(out_channels),
            nn.Conv3d(out_channels, out_channels, kernel_size=cfg.kernel_size, padding=cfg.padding),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)


class SingleConv(nn.Module):
    """(convolution => [IN] => LeakyReLU)
      Args:
          :param in_channels:(int) number of input channels
          :param out_channels: (int) number of output segmentation masks;
          :param stride_first_layer: (int) stride for the first conv layer
      """

    def __init__(self, in_channels, out_channels, stride_first_layer):
        super().__init__()
        # D_out=((D_in+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/stride[0])+1

        self.Single_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=cfg.kernel_size, padding=cfg.padding,
                      stride=stride_first_layer),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.Single_conv(x)
