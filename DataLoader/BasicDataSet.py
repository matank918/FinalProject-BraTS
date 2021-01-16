import os
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom
from matplotlib import pyplot
from sklearn import preprocessing
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
        """:return (int): check the length of the list "folder_names"""
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
                mask_obj = nib.load(file).get_fdata()
                self.seg_image = zoom(mask_obj, (0.535, 0.535, 0.825))
                self.seg_image = torch.from_numpy(self.seg_image)
            else:
                image_obj = nib.load(file).get_fdata()
                image_resized = zoom(image_obj, (0.535, 0.535, 0.825))
                img_list.append(image_resized)

        self.mri_image = np.stack(img_list, axis=0)
        self.mri_image = torch.from_numpy(self.mri_image)
        return self.mri_image, self.seg_image

    def show_image(self, i):

        """ display the differ angles of each image
                :param i (int):"""
        file_dir = self.folder_names[i]
        files_list = [os.path.join(file_dir, dI) for dI in os.listdir(file_dir)]
        for file in files_list:
            if 'seg' in file:
                pass
            else:
                image_obj = nib.load(file).get_fdata()
                fig, ax = plt.subplots(2, 2, figsize=(30, 30))
                ax[0, 0].imshow(image_obj[image_obj.shape[0] // 2])
                ax[0, 1].imshow(image_obj[:, image_obj.shape[1] // 2])
                ax[1, 0].imshow(image_obj[:, :, image_obj.shape[2] // 2])
                plt.show()

    def histogram_image(self, i):
        file_dir = self.folder_names[i]
        files_list = [os.path.join(file_dir, dI) for dI in os.listdir(file_dir)]
        for file in files_list:
            if 'seg' in file:
                pass
            else:
                image = sitk.ReadImage(file)
                result = sitk.GetArrayFromImage(image)
                # print(type(result))
                # print(result.shape)
                plt.figure('historgram')
                result = result.flatten()
                n, bins, patches = plt.hist(result, bins=256, range=(1, result.max()), facecolor='red',
                                                    alpha=0.75, histtype='step')
                plt.show()

    def norm_image(self, i):
        time_now=time.time()
        file_dir = self.folder_names[i]
        files_list = [os.path.join(file_dir, dI) for dI in os.listdir(file_dir)]
        for file in files_list:
            if 'seg' in file:
                pass
            else:
                image= nib.load(file).get_fdata()
                # scalers = {}
                # for i in range(image.shape[1]):
                #     scalers[i] = preprocessing.StandardScaler()
                #     image[i, :, :] = scalers[i].fit_transform(image[i, :, :])
                #     image[i,:,:] += image[i, :, :]

                # for j in range(image.shape[1]):
                #     scalers[j] = preprocessing.StandardScaler()
                #     image[:, j, :] = scalers[j].fit_transform(image[:, j, :])
                #     image += image[:, j, :]

                # for k in range(image.shape[2]):
                #     scalers[k] = preprocessing.StandardScaler()
                #     image[:, :, k] = scalers[k].fit_transform(image[:, :, k])
                #     image[:,:,k] += image[:, :, k]


                # norm_image = preprocessing.StandardScaler().fit_transform(image)
                # print(type(norm_image))
                # print(norm_image.shape)
                plt.figure('historgram_norm')
                result = image.flatten()
                n, bins, patches = plt.hist(result, bins=256, range=(1, result.max()), facecolor='red',
                                                    alpha=0.75, histtype='step')
                plt.show()
                print(time.time()-time_now)


if __name__ == '__main__':
    dir = r'C:\Users\Elinoy\Documents\project\new data'
    Dataset = BasicDataset(dir)
    Dataset.__getitem__(0)
    # Dataset.show_image(0)
    # Dataset.histogram_image(0)
    Dataset.norm_image(0)
