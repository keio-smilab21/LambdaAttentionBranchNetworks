import torch
from torch import nn
from torchinfo import summary

from models.lambda_layer import LambdaLayer
from models.lambda_resnet import conv1x1, Bottleneck
from models.convnext_block import LayerNorm, ConvNeXt_block


class CNNModel(nn.Module):
    def __init__(self, num_channel: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            num_channel, 16, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        downsample = nn.Sequential(
            conv1x1(16, 8 * 4, 1),
            nn.BatchNorm2d(8 * 4),
        )
        self.layer1 = Bottleneck(
            16, 8, 1, downsample=downsample, norm_layer=nn.BatchNorm2d, size=28
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

class model_CNN_3(nn.Module):
    def __init__(self, num_channels: int, num_classes: int=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(
            num_channels, 16, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        downsample = nn.Sequential(
            conv1x1(16, 8 * 4, 1),
            nn.BatchNorm2d(8 * 4),
        )
        self.layer1 = Bottleneck(
            16, 8, 1, downsample=downsample, norm_layer=nn.BatchNorm2d, size=28
        )

        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(64, 128)
        # self.relu4 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        # x = self.conv3(x)
        # x = self.relu3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        # x = self.fc1(x)
        # x = self.relu4(x)
        x = self.fc2(x)

        return x

def main():
    model = model_CNN_3(num_channels=3, num_classes=2)
    summary(model, input_size=(32, 3, 512, 512))

if __name__=="__main__":
    main()