import os
import torch
from torch.utils.data import Dataset
import torchvision
from DataLoader.BasicTransformations import *
from DataLoader.GeometricTransformations import *
import torchio as tio

from torch.utils.data import DataLoader, random_split
import numpy as np


def get_loaders(dataset, val_percent, batch_size):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return train_loader, val_loader


class CustomDataset(Dataset):
    def __init__(self, data_dir, transforms=(), data_transform=()):

        self.dir = data_dir
        self.training_dir = data_dir
        self.training_data = []
        self.folder_names = []
        self.load_data = LoadData((240, 240, 155))

        self.transform = Compose([
            ToTensor(),
            # RandomFlip(0.2),
            RandomCrop3D((240, 240, 155), (128, 128, 128)),
            *transforms
        ])

        self.data_transform = torchvision.transforms.Compose([
            *data_transform,
            CustomNormalize()
            ])

        self.get_folder_names()

    def get_folder_names(self):
        """create a list of paths for each MRI image's folder"""
        self.folder_names = [os.path.join(self.training_dir, dI) for dI in os.listdir(self.training_dir) if
                             os.path.isdir(os.path.join(self.training_dir, dI))]

    def __len__(self):
        """:return (int): return length of the list "folder_names"""
        return len(self.folder_names)

    def __getitem__(self, i):
        data, seg = self.load_data(self.folder_names[i])
        data, seg = self.transform(data, seg)
        data = self.data_transform(data)

        return data, seg


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, seg):
        for t in self.transforms:
            img, seg = t(img, seg)
        return img, seg

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


if __name__ == '__main__':
    dir = r"/home/kachel/MICA BraTS2020/"
    # transformations = transforms.Compose([ToTensor()])
    Dataset = CustomDataset(dir)
    print(Dataset.__len__())
    data, label = Dataset.__getitem__(5)
