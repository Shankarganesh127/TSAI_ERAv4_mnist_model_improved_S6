
# model_v1.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from data_setup import DataSetup
import logging


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 10, 3, padding=0, bias=False), 
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=0, bias=False), 
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding=0, bias=False), 
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(2,2), 
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(20, 10, 1, padding=0, bias=False), 
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, padding=0, bias=False), 
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, 3, padding=0, bias=False), 
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout2d(0.05),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 10, 1, padding=0, bias=False), 
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 10, 7, padding=0, bias=False), 
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)
    
    
class set_config_v1:
    def __init__(self):
        self.epochs = 15
        self.nll_loss = torch.nn.NLLLoss()  # Define NLLLoss here
        self.criterion = self.nll_loss
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()
        
    def setup(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.1)
        self.device = next(model.parameters()).device  # Get device from model
        self.dataloader_args = self.get_dataloader_args()
        self.data_setup_instance = DataSetup(**self.dataloader_args)
        logging.getLogger().info(f"Model v1 setup: Using SGD optimizer with lr=0.05, momentum=0.9, StepLR scheduler with step_size=6, gamma=0.1 for {self.epochs} epochs")
        logging.getLogger().info(f"Dataloader arguments: {self.dataloader_args}")
        return self
        
    def get_dataloader_args(self):
        data_loaders_args = {}
        if hasattr(self, 'device') and self.device.type == "cuda":
            data_loaders_args = dict(batch_size_train=128, batch_size_test=1000, shuffle_train=True, shuffle_test=False, 
                                   num_workers=2, pin_memory=True, train_transforms=self.train_transforms, test_transforms=self.test_transforms)
        else:
            data_loaders_args = dict(batch_size_train=128, batch_size_test=1000, shuffle_train=True, shuffle_test=False, 
                                   train_transforms=self.train_transforms, test_transforms=self.test_transforms)
        logging.info(f"Model v1 dataloader args: {data_loaders_args}")
        return data_loaders_args
        
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