
# model_v2.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from data_setup import DataSetup
import logging

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Efficient Initial Feature Extraction (4 channels)
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),  # -> 8x28x28
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(8, 10, 3, padding=1, bias=False),  # -> 10x28x28
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> 8x14x14
        )

        # Focused Feature Learning (10 channels)
        self.feature_block = nn.Sequential(
            nn.Conv2d(10, 10, 3, padding=1, bias=False),  # -> 10x14x14
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),  # -> 16x14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # -> 12x7x7
        )

        # Pattern Recognition 
        self.pattern_block = nn.Sequential(
            nn.Conv2d(16, 10, 1, bias=False),  # -> 8x7x7 (dimensionality reduction)
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(10, 16, 3, padding=1, bias=False),  # -> 16x7x7
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # Efficient Classification Head
        self.classifier = nn.Sequential(
            nn.Conv2d(16, 10, 3, bias=False),  # -> 10x5x5
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(0.05),
            nn.Conv2d(10, 16, 3, bias=False),  # -> 16x3x3
            nn.AvgPool2d(kernel_size=3),  # -> 16x1x1
            nn.Conv2d(16, 10, 1, bias=False),  # -> 16x1x1
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.input_conv(x)

        # Feature learning
        x = self.feature_block(x)

        # Pattern recognition with residual-like behavior
        x = self.pattern_block(x)
        
        # Classification
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)
    
class set_config_v3:
    def __init__(self):
        self.epochs = 15
        self.nll_loss = torch.nn.NLLLoss()  # Define NLLLoss here
        self.criterion = self.nll_loss
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()
        
    def setup(self, model):
        self.optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.1)
        self.device = next(model.parameters()).device  # Get device from model
        self.dataloader_args = self.get_dataloader_args()
        self.data_setup_instance = DataSetup(**self.dataloader_args)
        logging.getLogger().info(f"Model v3 setup: Using SGD optimizer with lr=0.01, momentum=0.9, StepLR scheduler with step_size=4, gamma=0.1 for {self.epochs} epochs")
        logging.getLogger().info(f"Dataloader arguments: {self.dataloader_args}")
        return self
        
    def get_dataloader_args(self):
        data_loaders_args = {}
        if hasattr(self, 'device') and self.device.type == "cuda":
            data_loaders_args = dict(batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False, 
                                   num_workers=2, pin_memory=True, train_transforms=self.train_transforms, test_transforms=self.test_transforms)
        else:
            data_loaders_args = dict(batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False, 
                                   train_transforms=self.train_transforms, test_transforms=self.test_transforms)
        logging.info(f"Model v3 dataloader args: {data_loaders_args}")
        return data_loaders_args
        
    def get_train_transforms(self):
        train_transforms = transforms.Compose([
            transforms.RandomRotation((-5.0, 5.0), fill=(0,)),
            transforms.RandomAffine(0, translate=(0.05, 0.05)),
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