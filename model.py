import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=True)

        for param in self.resnet18.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet18.layer2.parameters():
            param.requires_grad = False

        modules = list(self.resnet18.children())[:-1]
        self.resnet18 = nn.Sequential(*modules)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.resnet18(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x