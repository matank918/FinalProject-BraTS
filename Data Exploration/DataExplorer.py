import torch
import matplotlib.pyplot as plt
from utils.utils import split_image
from DataLoader.CustomDataset import CustomDataset
from torch.utils.data import DataLoader, random_split
import nibabel as nib
from DataLoader.BasicTransformations import train_transforms, val_transform
from skimage import data, color, io, img_as_float
import numpy as np
from monai.metrics import DiceMetric


def show_image(image, name=None):
    """display 3d image with shape of (128,128,128)"""
    # image = torch.squeeze(image)
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


def display_image(image, seg, slice_i):
    # pick one image from DecathlonDataset to visualize and check the 4 channels
    fig, axs = plt.subplots(3, 4, figsize=(20, 20))
    axs[0, 0].imshow(image[0, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 0].set_title("channel 1 - Sagittal")
    axs[1, 0].imshow(image[0, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 0].set_title("channel 1 - Coronal")
    axs[2, 0].imshow(image[0, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 0].set_title("channel 1 - Horizontal")
    axs[0, 1].imshow(image[1, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 1].set_title("channel 2 - Sagittal")
    axs[1, 1].imshow(image[1, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 1].set_title("channel 2 - Coronal")
    axs[2, 1].imshow(image[1, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 1].set_title("channel 2 - Horizontal")
    axs[0, 2].imshow(image[2, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 2].set_title("channel 3 - Sagittal")
    axs[1, 2].imshow(image[2, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 2].set_title("channel 3 - Coronal")
    axs[2, 2].imshow(image[2, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 2].set_title("channel 3 - Horizontal")
    axs[0, 3].imshow(image[3, slice_i, :, :].detach().cpu(), cmap="gray")
    axs[0, 3].set_title("channel 4 - Sagittal")
    axs[1, 3].imshow(image[3, :, slice_i, :].detach().cpu(), cmap="gray")
    axs[1, 3].set_title("channel 4 - Coronal")
    axs[2, 3].imshow(image[3, :, :, slice_i].detach().cpu(), cmap="gray")
    axs[2, 3].set_title("channel 4 - Horizontal")

    plt.show()
    if seg is not None:
        fig, axs = plt.subplots(3, 3, figsize=(20, 20))
        axs[0, 0].imshow(seg[0, slice_i, :, :].detach().cpu(), cmap="gray")
        axs[0, 0].set_title("Tumor core - Sagittal")
        axs[1, 0].imshow(seg[0, :, slice_i, :].detach().cpu(), cmap="gray")
        axs[1, 0].set_title("Tumor core - Coronal")
        axs[2, 0].imshow(seg[0, :, :, slice_i].detach().cpu(), cmap="gray")
        axs[2, 0].set_title("Tumor core - Horizontal")
        axs[0, 1].imshow(seg[1, slice_i, :, :].detach().cpu(), cmap="gray")
        axs[0, 1].set_title("WT - Sagittal")
        axs[1, 1].imshow(seg[1, :, slice_i, :].detach().cpu(), cmap="gray")
        axs[1, 1].set_title("WT - Coronal")
        axs[2, 1].imshow(seg[1, :, :, slice_i].detach().cpu(), cmap="gray")
        axs[2, 1].set_title("WT - Horizontal")
        axs[0, 2].imshow(seg[2, slice_i, :, :].detach().cpu(), cmap="gray")
        axs[0, 2].set_title("Enhancing tumor - Sagittal")
        axs[1, 2].imshow(seg[2, :, slice_i, :].detach().cpu(), cmap="gray")
        axs[1, 2].set_title("Enhancing tumor - Coronal")
        axs[2, 2].imshow(seg[2, :, :, slice_i].detach().cpu(), cmap="gray")
        axs[2, 2].set_title("Enhancing tumor - Horizontal")

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


def seg_mask(img, mask, alpha=0.6):
    img = img_as_float(img)
    rows, cols = img.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask > 0.9] = [1, 0, 0]  # Red block
    # color_mask[170:270, 40:120] = [0, 1, 0]  # Green block
    # color_mask[200:350, 200:350] = [0, 0, 1]  # Blue block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                      subplot_kw={'xticks': [], 'yticks': []})
    ax0.imshow(img, cmap="gray")
    ax1.imshow(color_mask)
    ax2.imshow(img_masked)
    plt.show()


if __name__ == '__main__':

    loader_path = r'C:\Users\User\Documents\FinalProject\MICCAI_BraTS2020\MICCAI_BraTS2020_TrainingData'
    # loader_path = '/tcmldrive/shared/BraTS2020 Training/'
    filename = r"C:\Users\User\AppData\Local\Temp\tmp5_bqwbm6\Task01_BrainTumour\imagesTr\BRATS_001.nii.gz"
    label = r"C:\Users\User\AppData\Local\Temp\tmp5_bqwbm6\Task01_BrainTumour\labelsTr\BRATS_001.nii.gz"
    # images = nib.load(filename)
    # label = nib.load(label).get_fdaya

    dataset = CustomDataset(data_dir=loader_path)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train.dataset.transform = train_transforms
    val.dataset.transform = val_transform
    # data = val.__getitem__(17)
    # images, seg = data["image"], data["seg"]
    # display_image(images, seg, 120)
    # show_image(seg)
    train_loader = DataLoader(train, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    # batch = iter(train_loader).next()
    # image, seg = batch["image"], batch["seg"]
    # tc = seg[:,0:1]
    # print(torch.unique(tc))
    # nans = torch.isnan(tc)

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
