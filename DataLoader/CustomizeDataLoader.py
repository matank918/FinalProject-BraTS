from torch.utils.data import DataLoader, random_split
from BasicDataSet import BasicDataset

class CustomizeDataLoader:

    def __init__(self):
        self.imgs_dir = 'data/carvana-image-masking-challenge/train/'
        self.load_data()

    def load_data(self, batch_size=1, val_percent=0.1):
        self.dataset = BasicDataset(self.imgs_dir)
        n_val = int(len(self.dataset) * val_percent)
        n_train = len(self.dataset) - n_val
        train, val = random_split(self.dataset, [n_train, n_val])
        self.train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        self.val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    def get_train_data(self):
        return self.train_loader

    def get_val_data(self):
        return self.val_loader

if __name__ == '__main__':
    Data = CustomizeDataLoader()
    train_loader = Data.get_train_data()
    l= Data.dataset.__getitem__(0)['image']
    for data in train_loader:
        k = data['image']
        break
    # print(len(train_loader))