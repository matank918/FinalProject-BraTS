import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import numpy as np
import time


class BasicDataset(Dataset):

    def __init__(self, images_dir):
        """:param images_dir: (str)"""

        self.dir = images_dir  # str
        self.folder_names = []  # list
        self.cancer_type = ['HGG', 'LGG']
        self.get_folder_names()

    def get_folder_names(self):
        """create a list of paths for each MRI image's folder"""
        for cancer_type in self.cancer_type:
            file_dir = os.path.join(self.dir, cancer_type)
            self.folder_names = self.folder_names + [os.path.join(file_dir, dI) for dI in os.listdir(file_dir) if
                                                     os.path.isdir(os.path.join(file_dir, dI))]

    def __len__(self):
        """:return (int): return length of the list "folder_names"""
        return len(self.folder_names)

    def __getitem__(self, i):
        """
        :param i (int):
        This function:
            1. load image
            2. resize
            3. join images
            4. convert to torch
        :return self.mri_image (torch.Tensor)
            self.seg_image: (torch.Tensor)"""

        file_dir = self.folder_names[i]
        img_list = []
        files_list = [os.path.join(file_dir, dI) for dI in os.listdir(file_dir)]
        for file in files_list:
            name = file[-12:-7]
            if 'seg' in file:
                """In seg file
                Label 1: necrotic and non-enhancing tumor
                Label 2: edema 
                Label 4: enhancing tumor
                Label 0: background
                """
                mask = nib.load(file).get_fdata()
                seg_image = zoom(mask, (0.535, 0.535, 0.825), mode='nearest')
                seg_image = self.quantize(seg_image)
                # seg_one_hot = self.encode_seg(seg_image)
                # ch0, ch1, ch2, ch4 = self.split_channels(seg_one_hot)
                # self.show_image(seg_image, name)
                # self.show_image(np.reshape(ch0, (128, 128, 128)), 'seg0')
                # self.show_image(np.reshape(ch1, (128, 128, 128)), 'seg1')
                # self.show_image(np.reshape(ch2, (128, 128, 128)), 'seg2')
                # self.show_image(np.reshape(ch4, (128, 128, 128)), 'seg4')

            else:
                image = nib.load(file).get_fdata()
                image_resized = zoom(image, (0.535, 0.535, 0.825))
                image_norm = self.norm_image(image_resized)
                # self.show_image(image_norm, name)
                # self.histogram_image(image_norm)
                img_list.append(image_norm)

        self.mri_image = torch.from_numpy(np.stack(img_list, axis=0))

        self.seg_image = torch.unsqueeze(torch.from_numpy(seg_image), 0)

        self.mri_image = self.mri_image.type(torch.float32)
        self.seg_image = self.seg_image.type(torch.float32)

        return {'mri_image': self.mri_image, 'seg': self.seg_image}

    def quantize(self, data):
        data[data < 0.5] = 0
        data[(0.5 < data) & (data < 1.5)] = 1
        data[(1.5 < data) & (data < 3)] = 2
        data[3 < data] = 4
        return data

    def encode_seg(self, data):
        values = np.array([0, 1, 2, 4])
        one_hot = np.zeros((4, 128, 128, 128))
        for i, value in enumerate(values):
            one_hot[i] = (data == value).astype(int)
        return one_hot

    @staticmethod
    def show_image(image, name):
        """display 3d image with shape of (128,128,128)"""
        fig = plt.figure()
        plt.title(name)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)

        ax1.title.set_text("Sagittal Section")
        ax1.imshow(image[image.shape[0] // 2], cmap=plt.gray())
        ax2.title.set_text("Coronal  Section")
        ax2.imshow(image[:, image.shape[1] // 2], cmap=plt.gray())
        ax3.title.set_text("Horizontal Section")
        ax3.imshow(image[:, :, image.shape[2] // 2], cmap=plt.gray())

        plt.show()

    @staticmethod
    def histogram_image(image):
        print("unique value:", np.unique(image))
        print("mean:", np.mean(image))
        print("var:", np.var(image))
        print("std:", np.std(image))

        plt.figure('historgram')
        result = image.flatten()
        plt.hist(result, bins=20, facecolor='red', alpha=0.75, histtype='step')
        plt.show()

    def norm_image(self, image):
        image = (image - np.mean(image)) / np.std(image)
        return image

    @staticmethod
    def split_channels(image):
        return np.split(image, 4, axis=0)


if __name__ == '__main__':
    dir = r"C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training"
    Dataset = BasicDataset(dir)
    item = Dataset.__getitem__(5)
    mri_image = item['mri_image']
    seg_image = item['seg']
