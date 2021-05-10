from torch.utils.data import DataLoader, random_split
import numpy as np
import collections.abc
import time
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union
import torch
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import os
import nibabel as nib
from monai.transforms import Orientationd, RandSpatialCropd, ToTensord, Compose, Randomizable, Transform, \
    apply_transform, NormalizeIntensityd


def get_loaders(dataset, val_percent, batch_size):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return train_loader, val_loader


class CustomDataset(_TorchDataset):

    def __init__(self, data_dir: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = self._generate_data(data_dir)
        self.transform = transform

    @staticmethod
    def _generate_data(data_dir) -> list:
        """create a list of paths for each MRI image's folder"""

        return [os.path.join(data_dir, dI) for dI in os.listdir(data_dir) if
                os.path.isdir(os.path.join(data_dir, dI))]

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        item = self.data[index]
        if not isinstance(self.transform, Compose):
            raise ValueError("transform must be an instance of monai.transforms.Compose.")

        for _transform in self.transform.transforms:
            item = apply_transform(_transform, item)
        return item

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index)


