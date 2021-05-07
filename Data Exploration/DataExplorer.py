import torch
import matplotlib.pyplot as plt
import torchvision
from DataLoader.BasicTransformations import *
from utils.utils import split_image
from DataLoader.CustomDataSet import CustomDataset, LoadData
from torchvision import transforms, datasets
from DataLoader.GeometricTransformations import *


def show_image(image, name=None):
    """display 3d image with shape of (128,128,128)"""
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
    # set all needed transformation
    # dir = r"/tcmldrive/databases/Public/MICA BRaTS2018/Training/HGG/Brats18_2013_2_1"
    # load = LoadData()
    # data, seg = load(file_dir=dir)

    # dir = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
    loader_path = '/tcmldrive/shared/BraTS2020 Training/'

    # transform_train = (RandomFlip(1,0.7),TranslateXYZ(1,0.7), Rotate(1,0.1), RandomElasticDeformation(1,0.1))
    # transform_target = (Rotate(1,0.5),)
    dataset = CustomDataset(data_dir=loader_path, transforms=())
    # dataset = CustomDataset(data_dir=dir, data_exploration=True)

    data, seg = dataset.__getitem__(17)

    mri1, mri2, mri3, mri4 = torch.chunk(data, dim=0, chunks=4)

    show_image(torch.squeeze(mri1))
    # show_image(torch.squeeze(mri2))
    # show_image(torch.squeeze(mri3))
    # show_image(torch.squeeze(mri4))

    _, ch1, ch2, ch4 = torch.chunk(seg, dim=0, chunks=4)

    # show_image(torch.squeeze(ch1), "ch1")
    show_image(torch.squeeze(ch2), "ch2")
    # show_image(torch.squeeze(ch4), "ch4")

