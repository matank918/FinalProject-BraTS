import torch
from torch import nn as nn
from utils.utils import flatten


class _AbstractDiceMetric(nn.Module):
    def __init__(self):
        super(_AbstractDiceMetric, self).__init__()

    def dice(self, input, target):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target)

        # average Dice score across all channels/classes
        return torch.mean(per_channel_dice)


class DiceMetric(_AbstractDiceMetric):
    def __init__(self):
        super().__init__()

    def dice(self, input, target):
        return self.compute_per_channel_dice(input, target)

    @staticmethod
    def compute_per_channel_dice(input, target, epsilon=1e-6):
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


def create_eval(name):
    if name == 'DiceMetric':
        return DiceMetric()
