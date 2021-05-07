import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchio as tio
import random

from abc import ABC, abstractmethod


class BaseMRItransform(ABC):

    def __init__(self, prob, mag=0):
        self.prob = prob
        self.mag = mag

    def __call__(self, img):
        rand = random.uniform(0, 1)
        if rand < self.mag:
            return self.transform(img)
        else:
            return img

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f)' % \
               (self.__class__.__name__, self.prob, self.mag)

    @abstractmethod
    def transform(self, img):
        pass


class RandomAnisotropy(BaseMRItransform):
    def transform(self, img):
        downsampling = (1, int(self.mag * 10))
        t = tio.transforms.RandomAnisotropy(downsampling=downsampling)
        return t(img)


class RandomMotion(BaseMRItransform):
    def transform(self, img):
        degrees = self.mag * 30
        translation = self.mag * 20
        t = tio.transforms.RandomMotion(degrees=degrees, translation=translation)
        return t(img)


class RandomGhosting(BaseMRItransform):
    def transform(self, img):
        intensity = self.mag * 2
        t = tio.transforms.RandomGhosting(intensity=intensity)
        return t(img)


class RandomSpike(BaseMRItransform):
    def transform(self, img):
        intensity = self.mag
        t = tio.transforms.RandomSpike(intensity=intensity)
        return t(img)


class RandomBiasField(BaseMRItransform):
    def transform(self, img):
        t = tio.transforms.RandomBiasField(coefficients=(-self.mag, self.mag), order=2)
        return t(img)


class RandomBlur(BaseMRItransform):
    def transform(self, img):
        kernel = int((img.shape[2] / 4) * self.mag)
        t = tio.transforms.RandomBlur(std=kernel)
        return t(img)


class RandomNoise(BaseMRItransform):
    def transform(self, img):
        t = tio.transforms.RandomNoise(mean=0, std=1)
        return t(img)


class RandomGamma(BaseMRItransform):
    def transform(self, img):
        t = tio.transforms.RandomGamma(log_gamma=(self.mag, -self.mag))
        return t(img)
