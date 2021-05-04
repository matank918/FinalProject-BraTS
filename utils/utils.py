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


def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.
    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)

    # cwd = pathlib.Path.cwd()
    # last_dir = cwd.name
    # cwd = str(cwd)
    # last_dir = str(last_dir)
    # imgs_dir = cwd.replace(last_dir, config['loader']['path'])
    # imgs_dir = imgs_dir + '\\' +'HGG'
