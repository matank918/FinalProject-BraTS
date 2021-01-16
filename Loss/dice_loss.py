import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


if __name__ == '__main__':
    img_rand = torch.rand(1, 1, 632, 951)
    mask_rand = torch.rand(1, 1, 632, 951)
    D = DiceLoss()
    print(D(img_rand,mask_rand))
    print(type(D(img_rand,mask_rand)))
