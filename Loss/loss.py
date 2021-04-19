import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from utils import flatten
import math


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target)

        # average Dice score across all channels/classes
        return 1 - torch.mean(per_channel_dice)

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
        # denominator = input.sum(-1) + target.sum(-1)
        denominator = torch.square(input).sum(-1) + torch.square(target).sum(-1)
        result = (intersect + epsilon) / (denominator + epsilon)
        return result


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf."""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target)

        # average Dice score across all channels/classes
        return 1 - torch.mean(per_channel_dice)

    def dice(self, input, target):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = flatten(input)
        target = flatten(target)
        target = target.float()

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())


class BCEDiceLoss(nn.Module):
    """Linear combination of BCE and Dice losses"""

    def __init__(self, alpha=0.5, beta=0.5):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.beta = beta
        self.dice = DiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


def create_loss(name):
    if name == 'DiceLoss':
        return DiceLoss()
    elif name == "BCEDiceLoss":
        return BCEDiceLoss()
    elif name == "GeneralizedDiceLoss":
        return GeneralizedDiceLoss()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rand_image1 = torch.empty(1, 4, 128, 128, 128, dtype=torch.float32).random_(2)
    rand_image2 = torch.empty(1, 4, 128, 128, 128, dtype=torch.float32).random_(2)
    loss = BCEDiceLoss(0.5,0.5)
    t = loss(rand_image1, rand_image1)
    print(t)
