import torch
import matplotlib.pyplot as plt
import torchvision
from DataLoader.BasicTransformations import *
from utils.utils import split_image
from DataLoader.CustomDataSet import CustomDataset
from torchvision import transforms, datasets
# from DataLoader.GeometricTransformations import *
from monai.transforms import Orientationd, RandSpatialCropd, ToTensord, Compose
from torch.utils.data import DataLoader, random_split


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


if __name__ == '__main__':
    image = r"C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData\BraTS20_Training_001"
    # load = LoadData((240, 240, 155))
    # data = load(file_dir=image)

    loader_path = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
    # loader_path = '/tcmldrive/shared/BraTS2020 Training/'

    dataset = CustomDataset(data_dir=loader_path, transform=train_transforms)

    data = dataset.__getitem__(95)

    mri1, mri2, mri3, mri4 = torch.chunk(data['image'], dim=0, chunks=4)

    show_image(mri1)
    show_image(mri2)
    show_image(mri3)
    show_image(mri4)

    show_image((data['seg']), "seg")
