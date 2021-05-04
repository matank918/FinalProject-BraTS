import torch
import matplotlib.pyplot as plt
import torchvision
from DataLoader.CustomTransformations import ToTensor, CustomNormalize, RandomCrop3D, OneHotEncoding3d
from utils.utils import split_image
from DataLoader.CustomDataSet import CustomDataset, LoadData
from torchvision import transforms, datasets


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
    dir = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData'
    transform_train = (
        transforms.RandomAffine(scale=(0.95, 1.05), degrees=8, shear=0.15, translate=(0.1, 0.1)),)
    dataset = CustomDataset(data_dir=dir, transforms=transform_train)
    data, seg = dataset.__getitem__(17)

    mri1, mri2, mri3, mri4 = torch.chunk(data, dim=0, chunks=4)
    mri1 = torch.squeeze(mri1)
    mri2 = torch.squeeze(mri2)
    mri3 = torch.squeeze(mri3)
    mri4 = torch.squeeze(mri4)


    show_image(mri1)
    show_image(mri2)
    show_image(mri3)
    show_image(mri4)

    _, ch1, ch2, ch4 = torch.chunk(seg, dim=0, chunks=4)
    ch1 = torch.squeeze(ch1)
    ch2 = torch.squeeze(ch2)
    ch4 = torch.squeeze(ch4)

    show_image(ch1, "ch1")
    show_image(ch2, "ch2")
    show_image(ch4, "ch4")

    Sagittal_ch1, Coronal_ch1, Horizontal_ch1 = split_image(ch1)

    Sagittal_ch1 = torch.squeeze(Sagittal_ch1)
    Coronal_ch1 = torch.squeeze(Coronal_ch1)
    Horizontal_ch1 = torch.squeeze(Horizontal_ch1)

    histogram_image(mri1)
