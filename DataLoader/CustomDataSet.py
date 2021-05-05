import os
import torch
from torch.utils.data import Dataset
import torchvision
from DataLoader.CustomTransformations import RandomCrop3D, OneHotEncoding3d, ToTensor, CustomNormalize
from torch.utils.data import DataLoader, random_split
import numpy as np
import nibabel as nib

def get_loaders(dataset, val_percent, batch_size):
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return train_loader, val_loader


class CustomDataset(Dataset):
    def __init__(self, data_dir, transforms=(), target_transform=()):
        """:param images_dir: (str)"""

        self.dir = data_dir
        self.training_dir = data_dir
        self.training_data = []
        self.folder_names = []
        self.load_data = LoadData()

        self.transforms = torchvision.transforms.Compose([
            ToTensor(),
            *transforms,
            CustomNormalize()])

        self.target_transform = torchvision.transforms.Compose([
            *target_transform,
            OneHotEncoding3d((240, 240, 155)),
            ToTensor()])

        self.get_folder_names()

    def get_folder_names(self):
        """create a list of paths for each MRI image's folder"""
        self.folder_names = [os.path.join(self.training_dir, dI) for dI in os.listdir(self.training_dir) if
                             os.path.isdir(os.path.join(self.training_dir, dI))]

    def __len__(self):
        """:return (int): return length of the list "folder_names"""
        return len(self.folder_names)

    def __getitem__(self, i):
        data, label = self.load_data(self.folder_names[i])

        data = self.transforms(data)
        label = self.target_transform(label)

        self.rand_crop = RandomCrop3D(data.shape, crop_dim=(128, 128, 128))

        data, label = self.rand_crop(data, label)
        return data, label


class LoadData(object):

    def __call__(self, file_dir, with_names=False):
        images = []
        images_names = []
        self.files_list = [os.path.join(file_dir, dI) for dI in os.listdir(file_dir)]
        for file in self.files_list:
            if 'seg' in file:
                seg = nib.load(file).get_fdata()
            else:
                images_names.append(file[-12:-7])
                image = (nib.load(file).get_fdata())
                images.append(image)

        images = np.stack(images, axis=0)

        if with_names:
            return images, seg, images_names
        else:
            return images, seg


if __name__ == '__main__':
    dir = r"/home/kachel/MICA BraTS2020/"
    # transformations = transforms.Compose([ToTensor()])
    Dataset = CustomDataset(dir)
    print(Dataset.__len__())
    data, label = Dataset.__getitem__(5)
