import torch
import matplotlib.pyplot as plt
import torchvision
from CustomTransformations import LoadData, ToTensor, CustomNormalize, RandomCrop3D, OneHotEncoding3d
from utils.utils import split_image, split_channels


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
    dir = r"/tcmldrive/databases/Public/MICA BRaTS2018/Training/HGG/Brats18_2013_2_1"
    load_data = LoadData()
    data, label, image_names = load_data(dir, with_names=True)
    data_transforms = torchvision.transforms.Compose([ToTensor(), CustomNormalize()])
    gt_transforms = torchvision.transforms.Compose([OneHotEncoding3d(label.shape), ToTensor()])

    net_dim = (128, 128, 128)
    rand_crop = RandomCrop3D(data.shape, net_dim)

    data = data_transforms(data)
    label = gt_transforms(label)

    data, label = rand_crop(data, label)

    mri1, mri2, mri3, mri4 = split_channels(data, dim=0)
    ch1, ch2, ch4 = split_channels(label, dim=0, chunk=3)
    show_image(mri1, image_names[0])
    show_image(mri2, image_names[1])
    show_image(mri3, image_names[2])
    show_image(mri4, image_names[3])
    show_image(ch1, "ch1")
    show_image(ch2, "ch2")
    show_image(ch4, "ch4")

    Sagittal_ch1, Coronal_ch1, Horizontal_ch1 = split_image(ch1)

    histogram_image(mri1)

