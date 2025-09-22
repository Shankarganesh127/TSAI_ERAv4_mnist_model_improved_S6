
# model_v0.py
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F

#
# Model definition (parameter-efficient, <20k params)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False), # -> 4x28x28
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, padding=1, bias=False), # -> 8x28x28
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # -> 8x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1, bias=False), # -> 16x14x14
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1, bias=False), # -> 32x14x14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2), # -> 32x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 40, 3, padding=1, bias=False), # -> 40x7x7
            nn.BatchNorm2d(40),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d(1) # -> 40x1x1
        self.fc = nn.Linear(40, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

