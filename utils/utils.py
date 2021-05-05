import logging
import numpy as np
import ast
import os
import torch
import shutil
from sklearn.model_selection import StratifiedShuffleSplit


def split_dataset(dataset, k):
    # load dataset
    X = list(range(len(dataset)))
    Y = dataset.targets

    # split to k-fold
    assert len(X) == len(Y)

    def _it_to_list(_it):
        return list(zip(*list(_it)))

    sss = StratifiedShuffleSplit(n_splits=k,  test_size=0.1)
    Dm_indexes, Da_indexes = _it_to_list(sss.split(X, Y))

    return Dm_indexes, Da_indexes


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


def split_image(image):
    Sagittal = torch.unsqueeze(image[image.shape[0] // 2], 0)
    Coronal = torch.unsqueeze(image[:, image.shape[1] // 2], 0)
    Horizontal = torch.unsqueeze(image[:, :, image.shape[2] // 2], 0)

    return Sagittal, Coronal, Horizontal


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])



