
# model_v3.py (aligned to model_v2 architecture, training tweaks only)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from data_setup import DataSetup
import logging

class Net(nn.Module):
    """Same layer structure and parameter count as model_v2.Net."""
    def __init__(self):
        super(Net, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.feature_block = nn.Sequential(
            nn.Conv2d(8, 10, 3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            #nn.Dropout2d(0.01),  # reduced dropout to improve train fit (no param change)
            nn.Conv2d(10, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.pattern_block = nn.Sequential(
            nn.Conv2d(16, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(16, 10, 1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 10, 3, bias=False),
            nn.AvgPool2d(kernel_size=5),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.feature_block(x)
        x = self.pattern_block(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class set_config_v3:
    """Training configuration tuned for higher train accuracy without adding parameters or altering layer structure."""
    def __init__(self):
        self.epochs = 15
        self.nll_loss = torch.nn.NLLLoss()
        self.criterion = self.nll_loss
        self.train_transforms = self.get_train_transforms()
        self.test_transforms = self.get_test_transforms()

    def setup(self, model, use_onecycle: bool = True):
        # Keep same model params; only training dynamics altered
        self.use_onecycle = use_onecycle
        base_lr = 0.065  # slightly higher for faster fit on small model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
        self.device = next(model.parameters()).device
        self.dataloader_args = self.get_dataloader_args()
        self.data_setup_instance = DataSetup(**self.dataloader_args)
        if self.use_onecycle:
            steps_per_epoch = len(self.data_setup_instance.train_loader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=base_lr,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,   # earlier peak
                div_factor=5,    # higher initial LR than before
                final_div_factor=100,
                anneal_strategy='cos'
            )
            # mark scheduler for per-batch stepping
            self.scheduler.batch_step = True
            logging.getLogger().info(
                f"Model v3: OneCycleLR max_lr={base_lr} pct_start=0.1 div_factor=5 final_div_factor=100 epochs={self.epochs}"
            )
        else:
            # Fallback StepLR (delayed decay)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=8, gamma=0.1)
            self.scheduler.batch_step = False
            logging.getLogger().info(
                f"Model v3: StepLR lr={base_lr} step_size=8 gamma=0.1"
            )
        logging.getLogger().info(f"Dataloader arguments: {self.dataloader_args}")
        return self

    def get_dataloader_args(self):
        # Reduce batch to 64 for more updates/epoch and slightly noisier grads (better fit)
        if hasattr(self, 'device') and self.device.type == "cuda":
            args = dict(batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False,
                        num_workers=2, pin_memory=True, train_transforms=self.train_transforms, test_transforms=self.test_transforms)
        else:
            args = dict(batch_size_train=64, batch_size_test=1000, shuffle_train=True, shuffle_test=False,
                        train_transforms=self.train_transforms, test_transforms=self.test_transforms)
        logging.info(f"Model v3 dataloader args: {args}")
        return args

    def get_train_transforms(self):
        # Further lighten augmentations to reduce underfitting pressure
        return transforms.Compose([
            transforms.RandomRotation((-5.0, 5.0), fill=(0,)),
            transforms.RandomAffine(0, translate=(0.03, 0.03)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def get_test_transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])