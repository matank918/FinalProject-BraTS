import torch
import os
import numpy as np
from torchvision import transforms
import torch.nn as nn
import nibabel as nib
from collections.abc import Iterable
from typing import Any, List, Optional, Sequence, Tuple, Union

from PIL import Image, ImageOps, ImageEnhance
import concurrent.futures
from monai.transforms import Orientationd, RandSpatialCropd, ToTensord, Compose, Randomizable, Transform, \
    apply_transform, NormalizeIntensityd, RandFlipd
from DataLoader.CustomTransformation import RandGaussianNoise3D, TranslateXYZ, Rotate3D, Scale3D


class LoadData(object):

    def __init__(self, seg_dim):
        self.values = np.array([0, 1, 2, 4])
        self.seg_dim = seg_dim

    def __call__(self, file_dir):

        images = []
        files_list = [os.path.join(file_dir, dI) for dI in os.listdir(file_dir)]
        for file in files_list:
            if 'seg' in file:
                seg = nib.load(file).get_fdata()
            else:
                image = (nib.load(file).get_fdata())
                images.append(image)
        images = np.stack(images, axis=0)
        seg = np.expand_dims(seg, axis=0).astype('float32')
        seg = np.where(seg == 4, 3, seg)
        return {"image": images, "seg": seg}


train_transforms = Compose([
    LoadData((240, 240, 155)),
    RandSpatialCropd(
        keys=["image", "seg"], roi_size=[128, 128, 128], random_size=False
    ),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    RandFlipd(keys=["image", "seg"], prob=0.2),
    # Rotate3D(keys=["image", "seg"], prob=1, mag=0),
    # RandGaussianNoise3D(keys=["image"], prob=0, mag=0.1),
    # TranslateXYZ(keys=["image", "seg"], prob=0, mag=1),
    # Scale3D(keys=["image", "seg"], prob=1, mag=1),
    ToTensord(keys=["image", "seg"])
])
