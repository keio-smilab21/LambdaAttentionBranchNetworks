from torch import nn

from models.lambda_layer import LambdaLayer
from models.lambda_resnet import conv1x1, Bottleneck


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
