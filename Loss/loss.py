import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


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
    """Computes Dice Loss according to https://arxiv.org/abs/1606.04797"""

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
        result = (intersect + epsilon) / (denominator + epsilon)
        return result


def create_loss(name):
    if name == 'DiceLoss':
        return DiceLoss()
    elif name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rand_image1 = torch.empty(1, 1, 128, 128, 128).random_(2)
    # rand_image2 = torch.empty(1, 1, 128, 128, 128).to(device)
    loss = DiceLoss()
    t = loss(rand_image1, rand_image1)
    print(t)