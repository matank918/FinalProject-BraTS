import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

def load_data(dir):
    files_list = [os.path.join(dir, dI) for dI in os.listdir(dir)]
    data = {}
    for file in files_list:
        name = file[-12:-7]
        seg_image = nib.load(file).get_fdata()
        data[name] = seg_image

    return data

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
    print("mean:", np.mean(image))
    print("var:", np.var(image))
    print("std:", np.std(image))

    plt.figure('historgram')
    result = image.flatten()
    plt.hist(result, bins=20, facecolor='red', alpha=0.75, histtype='step')
    plt.show()


if __name__ == '__main__':
    dir = r"C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training\HGG\Brats18_2013_2_1"
    data = load_data(dir)
    for image_name, image in data.items():
        show_image(image,image_name)
        histogram_image(image)