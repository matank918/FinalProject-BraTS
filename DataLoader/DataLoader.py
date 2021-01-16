from torch.utils.data import DataLoader, random_split
import pandas as pd
from BasicDataSet import BasicDataset
import numpy as np

class BasicDataloader:

    def __init__(self):
        self.imgs_dir = r'C:\Users\Elinoy\Documents\project\new data'
        self.load_data()
        self.data_dic = {"train_loader":[] ,"val_loader":[] }

    def load_data(self, batch_size=1, val_percent=0.1):
        self.dataset = BasicDataset(self.imgs_dir)
        print(self.dataset)
        print(len(self.dataset))
        n_val = int(len(self.dataset) * val_percent)
        n_train = len(self.dataset) - n_val
        train, val = random_split(self.dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        # self.data_dic["train_loader"] = self.train_loader
        self.val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
        # self.data_dic["val_loader"] = [DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)]
        # return self.data_dic
    # def get_train_data(self):
    #     return self.train_loader
    #
    # def get_val_data(self):
    #     return self.val_loader

if __name__ == '__main__':
    Dataloader =BasicDataloader()

