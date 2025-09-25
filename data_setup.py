
# data_setup.py
import torch.utils as utils
from torchvision import datasets, transforms

class DataSetup:
    def __init__(self, batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False, num_workers=2, pin_memory=None, train_transforms=None, test_transforms=None):
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transforms = train_transforms if train_transforms else self.get_train_transforms()
        self.test_transforms = test_transforms if test_transforms else self.get_test_transforms()
        self.train_loader = self.get_train_loader()
        self.test_loader = self.get_test_loader()

    def get_train_transforms(self):
        train_transforms = transforms.Compose([
            transforms.RandomRotation((-15.0, 15.0), fill=(0,)),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        return train_transforms
    
    def get_test_transforms(self):
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        return test_transforms

    def get_train_datasets(self):
        return datasets.MNIST('../data', train=True, download=True, transform=self.get_train_transforms())
    
    def get_test_datasets(self):
        return datasets.MNIST('../data', train=False, download=True, transform=self.get_test_transforms())
    
    def get_train_loader(self):
        train_dataset = self.get_train_datasets()
        return utils.data.DataLoader(train_dataset, batch_size=self.batch_size_train, shuffle=self.shuffle_train, num_workers=self.num_workers, pin_memory=self.pin_memory)
    
    def get_test_loader(self):
        test_dataset = self.get_test_datasets()
        return utils.data.DataLoader(test_dataset, batch_size=self.batch_size_test, shuffle=self.shuffle_test, num_workers=self.num_workers, pin_memory=self.pin_memory)

