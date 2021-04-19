import os
import torch
from torch.utils.data import Dataset
import torchvision
from DataLoader.CustomTransformations import RandomCrop3D, LoadData, OneHotEncoding3d, ToTensor, CustomNormalize


class CustomDataset(Dataset):
    def __init__(self, data_dir, transforms=None, data_dim=(240,240,155), net_dim=(128, 128, 128)):
        """:param images_dir: (str)"""

        self.dir = data_dir
        self.training_dir = data_dir
        # self.validation_dir = data_dir + '\\' + 'validation'
        self.cancer_type = ['HGG', 'LGG']
        self.training_data = []
        self.validation_data = []
        self.net_dim = net_dim
        self.folder_names = []
        self.transforms = transforms
        self.load_data = LoadData()

        if transforms is None:
            self.data_transforms = torchvision.transforms.Compose([ToTensor(), CustomNormalize()])
            self.gt_transforms = torchvision.transforms.Compose([OneHotEncoding3d(data_dim), ToTensor()])

        self.get_folder_names()

    def get_folder_names(self):
        """create a list of paths for each MRI image's folder"""
        for cancer_type in self.cancer_type:
            file_dir = os.path.join(self.training_dir, cancer_type)
            self.training_data = self.training_data + [os.path.join(file_dir, dI) for dI in os.listdir(file_dir) if
                                                     os.path.isdir(os.path.join(file_dir, dI))]

        # self.validation_data = [os.path.join(self.validation_dir, dI) for dI in os.listdir(self.validation_dir) if
        #                                              os.path.isdir(os.path.join(self.validation_dir, dI))]

        self.folder_names = self.training_data + self.validation_data

    def __len__(self):
        """:return (int): return length of the list "folder_names"""
        return len(self.folder_names)

    def __getitem__(self, i):
        data, label = self.load_data(self.folder_names[i])
        data = self.data_transforms(data)
        label = self.gt_transforms(label)

        self.rand_crop = RandomCrop3D(data.shape, self.net_dim)

        data, label = self.rand_crop(data,label)
        return data, label



if __name__ == '__main__':
    dir = r"/tcmldrive/databases/Public/MICA BRaTS2018"
    # transformations = transforms.Compose([ToTensor()])
    Dataset = CustomDataset(dir)
    data, label = Dataset.__getitem__(5)

