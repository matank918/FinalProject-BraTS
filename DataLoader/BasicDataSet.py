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
        self.get_folder_names()

    def get_folder_names(self):
        """create a list of paths for each MRI image's folder"""
        self.folder_names = [os.path.join(self.dir, dI) for dI in os.listdir(self.dir) if
                             os.path.isdir(os.path.join(self.dir, dI))]

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
            if 'seg' in file:
                """In seg file
                Label 1: necrotic and non-enhancing tumor
                Label 2: edema 
                Label 4: enhancing tumor
                Label 0: background
                """
                mask = nib.load(file).get_fdata()
                self.seg_image = zoom(mask, (0.535, 0.535, 0.825))
            else:
                image = nib.load(file).get_fdata()
                image_resized = zoom(image, (0.535, 0.535, 0.825))
                image_norm = self.norm_image(image_resized)
                #self.histogram_image(image_norm)
                img_list.append(image_norm)

        self.mri_image = np.stack(img_list, axis=0)
        self.seg_image = torch.unsqueeze(torch.from_numpy(self.seg_image),0)
        self.mri_image = torch.from_numpy(self.mri_image)
        return {'mri_image':self.mri_image, 'seg':self.seg_image}

    @staticmethod
    def show_image(image):
        """ display the differ angles of each image"""
        fig, ax = plt.subplots(2, 2, figsize=(30, 30))
        ax[0, 0].imshow(image[image.shape[0] // 2])
        ax[0, 1].imshow(image[:, image.shape[1] // 2])
        ax[1, 0].imshow(image[:, :, image.shape[2] // 2])
        plt.show()

    @staticmethod
    def histogram_image(image):
        print("mean:",np.mean(image))
        print("var:",np.var(image))
        print("std:",np.std(image))

        plt.figure('historgram')
        result = image.flatten()
        plt.hist(result, bins=20, facecolor='red', alpha=0.75, histtype='step')
        plt.show()

    def norm_image(self, image):
        image=(image-np.mean(image))/np.std(image)
        return image


if __name__ == '__main__':
    dir = r"C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training\HGG"
    Dataset = BasicDataset(dir)
    item = Dataset.__getitem__(0)
    mri_image = item['mri_image']
    seg_image = item['seg']