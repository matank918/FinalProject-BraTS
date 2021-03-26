import importlib
import torch
import torch.nn as nn
from UnetParts import Encoder, Decoder
from BuildingBlocks import DoubleConv, SingleConv
from utils import get_number_of_learnable_parameters, correct_type

class Abstract3DUNet(nn.Module):

    def __init__(self, in_channels, out_channels, f_maps, apply_pooling, interpolate,testing, basic_module=DoubleConv):
        """
          Args:
              :param in_channels:(int) number of input channels
              :param out_channels: (int) number of output segmentation masks;
              :param f_maps: (list) number of feature maps at each level of the encoder
              :param basic_module: (nn.Module) the base module for the net (default is DoubleConv)
              :param testing: (bool) if True we are at testing mode, if False we are at training mode
          """
        super(Abstract3DUNet, self).__init__()

        self.testing = testing

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = basic_module(in_channels, out_feature_num, stride_first_layer=1)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=basic_module, apply_pooling=apply_pooling)

            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        #  create decoder path consisting of the Decoder modules. The length of the decoder is equal to len(f_maps) - 1
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            conv_trans_channels = reversed_f_maps[i]
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]

            decoder = Decoder(conv_trans_channels, in_feature_num, out_feature_num, basic_module=basic_module,
                              interpolate=interpolate)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        # activation function
        if self.testing:
            self.final_activation = nn.Softmax(dim=1)

        # final Double conv
        self.final_conv = basic_module(f_maps[0], out_channels, 1)

    def forward(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            # print("encoder:", x.size())

        # remove the last encoder's output from the list
        encoders_features = encoders_features[1:]

        # decoder part
        for i, (decoder, encoder_features) in enumerate(zip(self.decoders, encoders_features)):
            # pass the output from the corresponding encoder and the output of the previous decoder
            x = decoder(encoder_features, x)
            if i > 0 and self.testing and self.final_activation is not None:
                x = self.final_activation(x)
            # print("decoder:", x.size())

        x = self.final_conv(x)

        if self.testing and self.final_activation is not None:
            x = self.final_activation(x)

        return x


class UNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, f_maps, apply_pooling, interpolate, testing, basic_module):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     f_maps=f_maps, apply_pooling=apply_pooling,
                                     interpolate=interpolate,testing=testing, basic_module=basic_module)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f_maps = [32, 64, 128, 256]
    model = UNet3D(4, 4, f_maps, interpolate=True,apply_pooling=False,testing=False)
    model.to(device)
    rand_image = torch.rand(1, 4, 128, 128, 128).to(device)
    print(get_number_of_learnable_parameters(model))
    (model(rand_image))
