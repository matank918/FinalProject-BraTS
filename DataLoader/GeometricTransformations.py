import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchio as tio
import random

from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageEnhance


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


class BaseGeometricTransform(ABC):

    def __init__(self, prob, mag=0):
        self.prob = prob
        self.mag = mag

    def __call__(self, img, seg):
        rand = random.uniform(0, 1)
        if rand < self.mag:
            return self.transform(img, seg)
        else:
            return img, seg

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
               (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img, seg):
        pass


class TranslateXYZ(BaseGeometricTransform):
    def transform(self, img, seg):
        translation = self.mag * 20
        t = tio.transforms.RandomAffine(translation=translation, image_interpolation='nearest')
        return t(img)


class Rotate(BaseGeometricTransform):

    def transform(self, img, seg):
        degrees = (int(self.mag * 30), int(self.mag * 30))
        t = tio.transforms.RandomAffine(degrees=degrees, image_interpolation='nearest')
        return t(img), t(seg)


class Scale(BaseGeometricTransform):
    def transform(self, img, seg):
        scale = self.mag * 0.5
        t = tio.transforms.RandomAffine(scales=scale, image_interpolation='nearest', isotropic=True)
        return t(img), t(seg)


class RandomElasticDeformation(BaseGeometricTransform):
    def transform(self, img, seg):
        displacement = self.mag*15
        t = tio.transforms.RandomElasticDeformation(max_displacement=displacement)
        return t(img)


class RandomAnisotropy(BaseTransform):
    def transform(self, img, seg):
        downsampling = (1, int(self.mag*10))
        t = tio.transforms.RandomAnisotropy(downsampling=downsampling)

        return t(img)


class RandomMotion(BaseTransform):
    def transform(self, img, seg):
        degrees = self.mag * 30
        translation = self.mag * 20
        t = tio.transforms.RandomMotion(degrees=degrees, translation=translation)
        return t(img)


class RandomGhosting(BaseTransform):
    def transform(self, img, seg):
        intensity = self.mag*2
        t = tio.transforms.RandomGhosting(intensity=intensity)
        return t(img)


class RandomSpike(BaseTransform):
    def transform(self, img, seg):
        intensity = self.mag
        t = tio.transforms.RandomSpike(intensity=intensity)
        return t(img)


class RandomBiasField(BaseTransform):
    def transform(self, img, seg):
        t = tio.transforms.RandomBiasField(coefficients=(-self.mag, self.mag), order=2)
        return t(img)


class RandomBlur(BaseTransform):
    def transform(self, img, seg):
        kernel = int((img.shape[2] / 4) * self.mag)
        t = tio.transforms.RandomBlur(std=kernel)
        return t(img)


class RandomNoise(BaseTransform):
    def transform(self, img, seg):
        t = tio.transforms.RandomNoise(mean=0, std=1)
        return t(img)


class RandomSwap(BaseTransform):
    def transform(self, img, seg):
        patch_size = (self.mag, self.mag, self.mag) * img.shape[2]
        t = tio.transforms.RandomSwap(patch_size=patch_size)
        return t(img)


class RandomGamma(BaseTransform):
    def transform(self, img, seg):
        t = tio.transforms.RandomGamma(log_gamma=(self.mag, -self.mag))
        return t(img)
