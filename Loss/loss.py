import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from utils import flatten
import math

class _AbstractDiceLoss(nn.Module):
    """Base class for different implementations of Dice loss."""

    def __init__(self):
        super(_AbstractDiceLoss, self).__init__()

    def dice(self, input, target):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target)

        # average Dice score across all channels/classes
        return 1 - torch.mean(per_channel_dice)


class DiceLoss(_AbstractDiceLoss):
    def __init__(self):
        super().__init__()

    def dice(self, input, target):
        return self.compute_per_channel_dice(input, target)

    @staticmethod
    def compute_per_channel_dice(input, target, epsilon=1e-6):
        """
        Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
        Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.
        Args:
             input (torch.Tensor): NxCxSpatial input tensor
             target (torch.Tensor): NxCxSpatial target tensor
             epsilon (float): prevents division by zero
        """

        # input and target shapes must match
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        # compute per channel Dice Coefficient
        intersect = 2 * (input * target).sum(-1)
        denominator = input.sum(-1) + target.sum(-1)
        # denominator = torch.square(input).sum(-1) + torch.square(target).sum(-1)
        result = (intersect + epsilon) / (denominator + epsilon)
        return result


class CustomBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def forward(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        loss = torch.zeros(input.shape[0], dtype=torch.float32)
        for i in range(loss.shape[0]):
            ones = torch.count_nonzero(target[i]).item()
            total = list(target[i].shape)[0]
            if ones != 0:
                weight = (total/ones)
            else:
                weight = 1000
            loss[i] = weight*self.loss(input[i],target[i])

        return torch.mean(loss)


def create_loss(name):
    if name == 'DiceLoss':
        return DiceLoss()
    elif name == "BCELoss":
        return CustomBCELoss()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rand_image1 = torch.empty(1, 4, 128, 128, 128, dtype=torch.float32).random_(2)
    rand_image2 = torch.empty(1, 4, 128, 128, 128, dtype=torch.float32).random_(2)
    loss = CustomBCELoss()
    t = loss(rand_image1, rand_image2)
    print(t)
