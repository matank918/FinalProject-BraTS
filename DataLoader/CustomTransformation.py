import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchio as tio
import random
from abc import ABC, abstractmethod
from math import pi
from monai.transforms import LoadImaged, CenterSpatialCropd, Compose, MapTransform, RandShiftIntensityd, \
    NormalizeIntensityd, RandGaussianNoised, RandFlipd, RandScaleIntensityd, Orientationd, ToTensord, \
    Spacingd, RandSpatialCropd, RandRotated, RandZoomd, RandAffined


class BaseTransform(ABC):

    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag

    def __call__(self, data):
        return self.transform(data)

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
               (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass


class RandGaussianNoise3D(BaseTransform):
    def transform(self, data):
        mean = self.mag * 5
        t = RandGaussianNoised(keys=["image"], prob=self.prob, mean=mean, std=1)
        return t(data)


class TranslateXYZ(BaseTransform):
    def transform(self, data):
        translation = (-int(self.mag * 20), int(self.mag * 20))
        t = RandAffined(keys=["image", "seg"], prob=self.prob, translate_range=translation, mode='nearest')
        return t(data)


class Rotate3D(BaseTransform):

    def transform(self, data):
        rotate_range = (-int(self.mag * 60), int(self.mag * 60))
        t = RandRotated(keys=["image", "seg"], prob=self.prob, range_x=rotate_range,
                        range_y=rotate_range, range_z=rotate_range)

        return t(data)


class Scale3D(BaseTransform):
    def transform(self, data):
        scale = (self.mag * 0.5, self.mag)
        t = RandAffined(keys=["image", "seg"], prob=self.prob, scale_range=scale, mode='nearest')
        return t(data)
#
#
# class RandomElasticDeformation(BaseGeometricTransform):
#     def transform(self, img, seg):
#         displacement = self.mag * 15
#         t = tio.transforms.RandomElasticDeformation(max_displacement=displacement)
#         return t(img), t(seg)
#
#
# class RandomSwap(BaseGeometricTransform):
#     def transform(self, img, seg):
#         patch_size = (self.mag, self.mag, self.mag) * img.shape[2]
#         t = tio.transforms.RandomSwap(patch_size=patch_size)
#         return t(img), t(seg)
#
#
# class RandomAnisotropy(BaseMRItransform):
#     def transform(self, img):
#         downsampling = (1, int(self.mag * 10))
#         t = tio.transforms.RandomAnisotropy(downsampling=downsampling)
#         return t(img)
#
#
# class RandomMotion(BaseMRItransform):
#     def transform(self, img):
#         degrees = self.mag * 30
#         translation = self.mag * 20
#         t = tio.transforms.RandomMotion(degrees=degrees, translation=translation)
#         return t(img)
#
#
# class RandomGhosting(BaseMRItransform):
#     def transform(self, img):
#         intensity = self.mag * 2
#         t = tio.transforms.RandomGhosting(intensity=intensity)
#         return t(img)
#
#
# class RandomSpike(BaseMRItransform):
#     def transform(self, img):
#         intensity = self.mag
#         t = tio.transforms.RandomSpike(intensity=intensity)
#         return t(img)
#
#
# class RandomBiasField(BaseMRItransform):
#     def transform(self, img):
#         t = tio.transforms.RandomBiasField(coefficients=(-self.mag, self.mag), order=2)
#         return t(img)
#
#
# class RandomBlur(BaseMRItransform):
#     def transform(self, img):
#         kernel = int((img.shape[2] / 4) * self.mag)
#         t = tio.transforms.RandomBlur(std=kernel)
#         return t(img)
#
#
# class RandomNoise(BaseMRItransform):
#     def transform(self, img):
#         t = tio.transforms.RandomNoise(mean=0, std=1)
#         return t(img)
#
#
# class RandomGamma(BaseMRItransform):
#     def transform(self, img):
#         t = tio.transforms.RandomGamma(log_gamma=(self.mag, -self.mag))
#         return t(img)
