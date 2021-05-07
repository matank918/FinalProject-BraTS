import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchio as tio
import random

from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageEnhance




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


class RandomFlip(BaseGeometricTransform):
    def transform(self, img, seg):
        t = tio.transforms.RandomFlip(flip_probability=1)
        return t(img), t(seg)


class TranslateXYZ(BaseGeometricTransform):
    def transform(self, img, seg):
        translation = self.mag * 20
        t = tio.transforms.RandomAffine(translation=translation, image_interpolation='nearest')
        return t(img), t(seg)


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
        return t(img), t(seg)


class RandomSwap(BaseGeometricTransform):
    def transform(self, img, seg):
        patch_size = (self.mag, self.mag, self.mag) * img.shape[2]
        t = tio.transforms.RandomSwap(patch_size=patch_size)
        return t(img), t(seg)



