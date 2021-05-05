import torch
import os
import numpy as np
from torchvision import transforms
import torch.nn as nn

from PIL import Image, ImageOps, ImageEnhance



class OneHotEncoding3d(object):
    def __init__(self, dim):
        self.values = np.array([0, 1, 2, 4])
        self.one_hot = np.zeros((len(self.values),) + dim)

    def __call__(self, seg):
        for i, value in enumerate(self.values):
            self.one_hot[i] = (seg == value).astype(int)
        return self.one_hot


class ToTensor(object):
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float32)


class RandomCrop3D(object):
    def __init__(self, img_dim, crop_dim):
        c, h, w, d = img_dim
        assert (h, w, d) > crop_dim
        self.img_dim = tuple((h, w, d))
        self.crop_dim = tuple(crop_dim)

    def __call__(self, data, label):
        assert data.shape[1:] == label.shape[1:]
        slice_hwd = [self._get_slice(i, k) for i, k in zip(self.img_dim, self.crop_dim)]
        return self._crop(data, *slice_hwd), self._crop(label, *slice_hwd)

    @staticmethod
    def _get_slice(img_dim, crop_dim):
        try:
            lower_bound = torch.randint(img_dim - crop_dim, (1,)).item()
            return lower_bound, lower_bound + crop_dim
        except:
            return (None, None)

    @staticmethod
    def _crop(x, slice_h, slice_w, slice_d):
        return x[:, slice_h[0]:slice_h[1], slice_w[0]:slice_w[1], slice_d[0]:slice_d[1]]


class CustomNormalize(object):

    def __call__(self, x):
        for i in range(x.shape[0]):
            non_zero_vox = x[i][x[i].nonzero(as_tuple=True)]
            x[i][x[i].nonzero(as_tuple=True)] = \
                ((x[i][x[i].nonzero(as_tuple=True)] - torch.mean(non_zero_vox)) / torch.std(non_zero_vox))

        return x




if __name__ == '__main__':
    a = (240, 240, 155)
    b = 4
    new = (b,) + a
    print(new)
