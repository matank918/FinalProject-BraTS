import os
from torch.utils.data import Dataset
from torchvision import transforms
from DataLoader.CustomTransformations import RandomCrop3D, LoadData, OneHotEncoding3d, ToTensor, CustomNormalize


class BasicDataset(Dataset):
    def __init__(self, data_dir, transforms=None, net_dim=(128, 128, 128)):
        """:param images_dir: (str)"""

        self.dir = data_dir
        self.training_dir = data_dir + '\\' + 'Training'
        # self.validation_dir = data_dir + '\\' + 'validation'
        self.cancer_type = ['HGG', 'LGG']
        self.training_data = []
        self.validation_data = []

        self.folder_names = []
        self.net_dim = net_dim
        self.transforms = transforms
        self.load_data = LoadData()
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
        data, label = self.basic_transform(data, label)
        return data, label

    def basic_transform(self, data, label):
        one_hot = OneHotEncoding3d(label.shape)
        rand_crop = RandomCrop3D(data.shape, self.net_dim)
        to_tensor = ToTensor()
        normalize = CustomNormalize()

        return rand_crop(normalize(to_tensor(data)), to_tensor(one_hot(label)))


if __name__ == '__main__':
    dir = r"C:\Users\User\Documents\FinalProject\MICA BRaTS2018"
    transformations = transforms.Compose([ToTensor()])
    Dataset = BasicDataset(dir, transformations)
    data, label = Dataset.__getitem__(5)

