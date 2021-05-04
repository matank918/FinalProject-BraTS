import torch
import torch.nn as nn
from nnUnet.UnetParts import Encoder, Decoder
from nnUnet.BuildingBlocks import DoubleConv
from utils.utils import get_number_of_learnable_parameters
import copy


class Abstract3DUNet(nn.Module):

    def __init__(self, in_channels, out_channels, f_maps, apply_pooling, deep_supervision, basic_module=DoubleConv):
        """
          Args:
              :param in_channels:(int) number of input channels
              :param out_channels: (int) number of output segmentation masks;
              :param f_maps: (list) number of feature maps at each level of the encoder
              :param basic_module: (nn.Module) the base module for the net (default is DoubleConv)
              :param deep_supervision: (int) apply deep supervision for number of layer.
          """
        self.deep_supervision = deep_supervision

        super(Abstract3DUNet, self).__init__()

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i in range(len(f_maps)):
            if i == 0:
                encoder = basic_module(in_channels, f_maps[i], stride_first_layer=1)
            else:
                encoder = Encoder(f_maps[i - 1], f_maps[i], basic_module=basic_module, apply_pooling=apply_pooling)

            encoders.append(encoder)
        self.encoders = nn.ModuleList(encoders)

        #  create decoder path consisting of the Decoder modules. The length of the decoder is equal to len(f_maps) - 1
        decoders = []
        output_activation_layers = []
        reversed_f_maps = list(reversed(f_maps))[1:]
        reversed_f_maps.append(out_channels)
        for i in range(1, len(reversed_f_maps)):
            in_feature_num = reversed_f_maps[i - 1]
            out_feature_num = reversed_f_maps[i]

            decoder = Decoder(in_feature_num, out_feature_num, basic_module=basic_module)
            output_activation = nn.Sequential(nn.Conv3d(out_feature_num, out_channels, kernel_size=(1, 1, 1)),
                                              nn.Softmax(dim=1))
            decoders.append(decoder)
            output_activation_layers.append(output_activation)

        self.decoders = nn.ModuleList(decoders)
        self.output_activation = nn.ModuleList(output_activation_layers)

        print(self.decoders)
        print(self.output_activation)

    def forward(self, x):
        # encoder part
        input_dim = (128, 128, 128)
        encoders_features = []
        decoders_features = []
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
            _, _, D, W, H = x.shape
            # no activation layer for first layer
            if i > (3 - self.deep_supervision):
                output_activation = self.output_activation[i]
                encoders_feat = output_activation(x)
                if (D, W, H) != input_dim:
                    encoders_feat = nn.functional.interpolate(encoders_feat, size=input_dim, mode='trilinear')
                decoders_features.append(encoders_feat)

        return decoders_features


class UNet3D(Abstract3DUNet):
    def __init__(self, in_channels, out_channels, f_maps, apply_pooling, basic_module, deep_supervision):
        super(UNet3D, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                     f_maps=f_maps, apply_pooling=apply_pooling, basic_module=basic_module,
                                     deep_supervision=deep_supervision)


if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f_maps = [32, 64, 128, 256, 320, 320]
    model = UNet3D(4, 4, f_maps, apply_pooling=False, basic_module=DoubleConv, deep_supervision=3)
    model.to(device)
    # x = model.training
    # rand_image = torch.rand(2, 4, 128, 128, 128).to(device)
    # print(get_number_of_learnable_parameters(model))
    # out = model(rand_image)
