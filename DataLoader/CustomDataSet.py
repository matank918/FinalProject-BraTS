import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from DataLoader.CustomTransformations import RandomCrop3D, LoadData, OneHotEncoding3d, ToTensor, CustomNormalize

class BasicDataset(Dataset):

    def __init__(self, images_dir, transforms=None, net_dim =(128,128,128)):
        """:param images_dir: (str)"""

        self.dir = images_dir
        self.net_dim = net_dim

        self.folder_names = []
        self.cancer_type = ['HGG', 'LGG']
        self.get_folder_names()
        self.transforms = transforms

        self.load_data = LoadData()

    def get_folder_names(self):
        """create a list of paths for each MRI image's folder"""
        for cancer_type in self.cancer_type:
            file_dir = os.path.join(self.dir, cancer_type)
            self.folder_names = self.folder_names + [os.path.join(file_dir, dI) for dI in os.listdir(file_dir) if
                                                     os.path.isdir(os.path.join(file_dir, dI))]

    def __len__(self):
        """:return (int): return length of the list "folder_names"""
        return len(self.folder_names)

    def __getitem__(self, i):
        data, label = self.load_data(self.folder_names[i])
        data, label = self.basic_transform(data, label)

        return data, label

    def basic_transform(self, data, label):
        one_hot = OneHotEncoding3d(np.unique(label),label.shape)
        rand_crop = RandomCrop3D(data.shape, self.net_dim)
        to_tensor = ToTensor()
        normalize = CustomNormalize()

        return rand_crop(normalize(to_tensor(data)), to_tensor(one_hot(label)))


if __name__ == '__main__':
    dir = r"C:\Users\User\Documents\FinalProject\MICA BRaTS2018\Training"
    transformations = transforms.Compose([ToTensor()])
    Dataset = BasicDataset(dir, transformations)
    data, label = Dataset.__getitem__(5)


    """In seg file
    Label 1: necrotic and non-enhancing tumor
    Label 2: edema 
    Label 4: enhancing tumor
    Label 0: background
    """
