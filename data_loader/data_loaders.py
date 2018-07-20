import torch
import numpy as np
from torchvision import datasets, transforms
from base import BaseDataLoader
import os

class ImageNetDataLoader(BaseDataLoader):
    """
    ImageNet data loader
    """
    def __init__(self, config):
        super(ImageNetDataLoader, self).__init__(config)
        self.data_dir = config['data_loader']['data_dir']
        self.batch_size = config['data_loader']['batch_size']
        self.shuffle = config['data_loader']['shuffle']
        self.data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(self.data_dir, 'train'),
                        transform=transforms.Compose([
                            transform.RandomResizedCrop(224),
                            transform.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), 
                                                 std=(0.229, 0.224, 0.225))
                           ])), batch_size=self.batch_size, 
                                shuffle=self.shuffle
        )
        self.x, self.y = [], []
        for data, target in self.data_loader:
            self.x += [i for i in data.numpy()]
            self.y += [i for i in target.numpy()]
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __next__(self):
        batch = super(ImageNetDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        packed = list(zip(self.x, self.y))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, packed):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, config):
        super(MnistDataLoader, self).__init__(config)
        self.data_dir = config['data_loader']['data_dir']
        self.data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=256, shuffle=False)
        self.x = []
        self.y = []
        for data, target in self.data_loader:
            self.x += [i for i in data.numpy()]
            self.y += [i for i in target.numpy()]
        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __next__(self):
        batch = super(MnistDataLoader, self).__next__()
        batch = [np.array(sample) for sample in batch]
        return batch

    def _pack_data(self):
        packed = list(zip(self.x, self.y))
        return packed

    def _unpack_data(self, packed):
        unpacked = list(zip(*packed))
        unpacked = [list(item) for item in unpacked]
        return unpacked

    def _update_data(self, unpacked):
        self.x, self.y = unpacked

    def _n_samples(self):
        return len(self.x)
