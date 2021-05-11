import torch
import matplotlib.pyplot as plt
import torchvision
from DataLoader.BasicTransformations import LoadData, nib, train_transforms
from utils.utils import split_image
from DataLoader.CustomDataset import CustomDataset
from torchvision import transforms, datasets
from monai.transforms import Orientationd, RandSpatialCropd, ToTensord, Compose
from torch.utils.data import DataLoader, random_split
import cv2 as cv
import numpy as np
from PIL import Image

def show_image(image, name=None):
    """display 3d image with shape of (128,128,128)"""
    image = torch.squeeze(image)
    fig = plt.figure()
    if name is not None:
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


def display_image(data, slice_i):
    # pick one image from DecathlonDataset to visualize and check the 4 channels
    fig, axs = plt.subplots(3, 4, figsize=(20, 20))
    axs[0, 0].imshow(data["image"][0, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 0].set_title("channel 1 - Sagittal")
    axs[1, 0].imshow(data["image"][0, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 0].set_title("channel 1 - Coronal")
    axs[2, 0].imshow(data["image"][0, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 0].set_title("channel 1 - Horizontal")
    axs[0, 1].imshow(data["image"][1, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 1].set_title("channel 2 - Sagittal")
    axs[1, 1].imshow(data["image"][1, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 1].set_title("channel 2 - Coronal")
    axs[2, 1].imshow(data["image"][1, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 1].set_title("channel 2 - Horizontal")
    axs[0, 2].imshow(data["image"][2, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 2].set_title("channel 3 - Sagittal")
    axs[1, 2].imshow(data["image"][2, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 2].set_title("channel 3 - Coronal")
    axs[2, 2].imshow(data["image"][2, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 2].set_title("channel 3 - Horizontal")
    axs[0, 3].imshow(data["image"][3, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 3].set_title("channel 4 - Sagittal")
    axs[1, 3].imshow(data["image"][3, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 3].set_title("channel 4 - Coronal")
    axs[2, 3].imshow(data["image"][3, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 3].set_title("channel 4 - Horizontal")

    plt.show()

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    axs[0, 0].imshow(data["seg"][0, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 0].set_title("channel 1 - Sagittal")
    axs[1, 0].imshow(data["seg"][0, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 0].set_title("channel 1 - Coronal")
    axs[2, 0].imshow(data["seg"][0, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 0].set_title("channel 1 - Horizontal")
    axs[0, 1].imshow(data["seg"][1, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 1].set_title("WT - Sagittal")
    axs[1, 1].imshow(data["seg"][1, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 1].set_title("WT - Coronal")
    axs[2, 1].imshow(data["seg"][1, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 1].set_title("WT - Horizontal")
    axs[0, 2].imshow(data["seg"][2, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 2].set_title("channel 3 - Sagittal")
    axs[1, 2].imshow(data["seg"][2, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 2].set_title("channel 3 - Coronal")
    axs[2, 2].imshow(data["seg"][2, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 2].set_title("channel 3 - Horizontal")

    plt.show()


def histogram_image(image):
    # print("unique value:", np.unique(image))
    print("mean:", torch.mean(image))
    print("var:", torch.var(image))
    print("std:", torch.std(image))

    print("number of nonzero voxels:", len(torch.nonzero(image)))
    print("mean for non-zero voxels:", torch.mean(image[image.nonzero(as_tuple=True)]))
    print("var for non zero voxels:", torch.var(image[image.nonzero(as_tuple=True)]))
    print("std for non zero voxels:", torch.std(image[image.nonzero(as_tuple=True)]))

    plt.figure('historgram')
    result = image.flatten()
    plt.hist(result, bins=20)
    plt.show()


def display_image_and_seg(data, slice_i):
    fig, axs = plt.subplots(3, 4, figsize=(20, 20))
    for i in range(3):
        for j in range(4):
            merged_image = add_image_and_seg(data["image"][i, slice_i, :, :], data["seg"][2, slice_i, :, :])
            axs[i, j].imshow(merged_image, cmap="gray")
    # axs[0, 0].set_title("channel 1 - Sagittal")
    # axs[1, 0].imshow(data["image"][0, :, slice_i, :].detach().cpu(), cmap="gray")
    # axs[1, 0].imshow(data["seg"][1, :, slice_i, :].detach().cpu(), ccmap="jet", alpha=0.5)
    # axs[1, 0].set_title("channel 1 - Coronal")
    # axs[2, 0].imshow(data["image"][0, :, :, slice_i].detach().cpu(), cmap="gray")
    # axs[2, 0].imshow(data["seg"][1, :, :, slice_i].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[2, 0].set_title("channel 1 - Horizontal")
    # axs[0, 1].imshow(data["image"][1, slice_i, :, :].detach().cpu(), cmap="gray")
    # axs[0, 1].imshow(data["seg"][2, slice_i, :, :].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[0, 1].set_title("channel 2 - Sagittal")
    # axs[1, 1].imshow(data["image"][1, :, slice_i, :].detach().cpu(), cmap="gray")
    # axs[1, 1].imshow(data["seg"][1, :, slice_i, :].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[1, 1].set_title("channel 2 - Coronal")
    # axs[2, 1].imshow(data["image"][1, :, :, slice_i].detach().cpu(), cmap="gray")
    # axs[2, 1].imshow(data["seg"][1, :, :, slice_i].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[2, 1].set_title("channel 2 - Horizontal")
    # axs[0, 2].imshow(data["image"][2, slice_i, :, :].detach().cpu(), cmap="gray")
    # axs[0, 2].imshow(data["seg"][2, slice_i, :, :].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[0, 2].set_title("channel 3 - Sagittal")
    # axs[1, 2].imshow(data["image"][2, :, slice_i, :].detach().cpu(), cmap="gray")
    # axs[1, 2].imshow(data["seg"][1, :, slice_i, :].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[1, 2].set_title("channel 3 - Coronal")
    # axs[2, 2].imshow(data["image"][2, :, :, slice_i].detach().cpu(), cmap="gray")
    # axs[2, 2].imshow(data["seg"][1, :, :, slice_i].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[2, 2].set_title("channel 3 - Horizontal")
    # axs[0, 3].imshow(data["image"][3, slice_i, :, :].detach().cpu(), cmap="gray")
    # axs[0, 3].imshow(data["seg"][2, slice_i, :, :].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[0, 3].set_title("channel 4 - Sagittal")
    # axs[1, 3].imshow(data["image"][3, :, slice_i, :].detach().cpu(), cmap="gray")
    # axs[1, 3].imshow(data["seg"][1, :, slice_i, :].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[1, 3].set_title("channel 4 - Coronal")
    # axs[2, 3].imshow(data["image"][3, :, :, slice_i].detach().cpu(), cmap="gray")
    # axs[2, 3].imshow(data["seg"][1, :, :, slice_i].detach().cpu(), cmap="jet", alpha=0.5)
    # axs[2, 3].set_title("channel 4 - Horizontal")

    plt.show()

def add_image_and_seg(image, seg):
    return cv.addWeighted(image, 0.9, seg, 0.1, 128)



if __name__ == '__main__':
    # file_path = r"C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001\BraTS20_Training_001_flair.nii.gz"
    # image = nib.load(file_path)
    # load = LoadData()
    # data = load(file_dir=image)

    loader_path = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
    # loader_path = '/tcmldrive/shared/BraTS2020 Training/'

    dataset = CustomDataset(data_dir=loader_path, transform=train_transforms)
    #
    data = dataset.__getitem__(5)
    display_image(data, 100)
    # image = data["image"][0, 100, :, :]
    # seg = data["seg"][1, :, 100, :]
    # fig = plt.figure()
    # plt.imshow(add_image_and_seg(image, seg))
    # plt.show()

    # for dat in dataset[10:20]:
    #     display_image(dat, 120)

    # print(torch.unique(data['seg']))
    # #
    # mri1, mri2, mri3, mri4 = torch.chunk(data['image'], dim=0, chunks=4)
    #
    # show_image(mri1)
    # show_image(mri2)
    # show_image(mri3)
    # show_image(mri4)

    # ch1, ch2, ch3 = torch.chunk(data['seg'], dim=0, chunks=3)

    # show_image(ch1)
    # show_image(ch2)
    # show_image(ch3)
