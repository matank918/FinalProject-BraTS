import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [IN] => LeakyReLU) * 2
      Args:
          :param in_channels:(int) number of input channels
          :param out_channels: (int) number of output segmentation masks;
          :param stride_first_layer: (int) stride for the first conv layer
      """

    def __init__(self, in_channels, out_channels, stride_first_layer):
        super().__init__()
        #D_out=((D_in+2×padding[0]−dilation[0]×(kernel_size[0]−1)−1)/stride[0])+1
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1,
                      stride=stride_first_layer),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


