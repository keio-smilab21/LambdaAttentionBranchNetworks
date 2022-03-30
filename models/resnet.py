"""
pytorch : torchvision.models.resnetの公式実装参考
"""
from torchinfo import summary
import torch
import torch.nn as nn

from lambda_layer import LambdaLayer

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1,
        bias=False
    )

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels,
        kernel_size=3, stride=stride, padding=1,
        bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1 # expansion = out_channel/in_channel

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, channels, stride)
        self. bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

        # in_channel != out_channel -> dawnsampling
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels*self.expansion, stride),
                nn.BatchNorm2d(channels*self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1):
        # print(in_channels, channels)
        super(Bottleneck, self).__init__()

        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels*self.expansion, stride),
                nn.BatchNorm2d(channels*self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out

class LambdaBasicBlock(nn.Module):
    expansion = 1 # expansion = out_channel/in_channel

    def __init__(self, in_channels, channels, stride=1, size=None):
        super(LambdaBasicBlock, self).__init__()

        self.conv1 = conv3x3(in_channels, channels, stride)
        # self.conv2 = LambdaLayer(channels, m=size, stride=stride)
        self. bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(channels, channels)
        self.conv2 = LambdaLayer(channels, m=size, stride=stride)
        self.bn2 = nn.BatchNorm2d(channels)

        # in_channel != out_channel -> dawnsampling
        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels*self.expansion, stride=1),
                nn.BatchNorm2d(channels*self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        self.q = self.conv2.q
        self.k = self.conv2.k
        # self.v = self.conv2.v
        self.yc = self.conv2.yc

        out = self.bn2(out)
        print("out : ", out.size())

        print("ssss", self.shortcut(x).size())
        out += self.shortcut(x)

        out = self.relu(out)

        return out


class LambdaBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, size=None):
        # print(in_channels, channels)
        super(LambdaBottleneck, self).__init__()

        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        # self.conv2 = conv3x3(channels, channels, stride)
        self.conv2 = LambdaLayer(channels, m=size, stride=stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels*self.expansion)
        self.bn3 = nn.BatchNorm2d(channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != channels * self.expansion:
            self.shortcut = nn.Sequential(
                conv1x1(in_channels, channels*self.expansion, stride),
                nn.BatchNorm2d(channels*self.expansion),
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        self.q = self.conv2.q
        self.k = self.conv2.k
        # self.v = self.conv2.v
        self.yc = self.conv2.yc

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv_2
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        # conv_3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # conv_4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # conv_5
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, channels, blocks, stride):
        layers = []

        # first residual layer
        layers.append(block(self.in_channels, channels, stride))

        # other residual layer
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class LambdaResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True, norm_layer=None):
        super(LambdaResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv_2
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, size=56)
        # conv_3
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, size=28)
        # conv_4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, size=14)
        # conv_5
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, size=7)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
    
    def _make_layer(self, block, channels, blocks, stride, size=None):
        norm_layer = self._norm_layer
        layers = []

        # first residual layer
        layers.append(block(self.in_channels, channels, stride, size))

        # other residual layer
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels, size=size))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


        

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(Bottleneck, [3,4,6,3])

def resnet101():
    return ResNet(Bottleneck, [3,4,23,3])

def resnet152():
    return ResNet(Bottleneck, [3,8,36,3])

def lambda_resnet18(**kwargs):
    return LambdaResNet(LambdaBasicBlock, [2,2,2,2], **kwargs)

def lambda_resnet34(**kwargs):
    return LambdaResNet(LambdaBasicBlock, [3,4,6,3], **kwargs)

def lambda_resnet38(**kwargs):
    return LambdaResNet(LambdaBottleneck, [2,3,5,2], **kwargs)

def lambda_resnet50(**kwargs):
    return LambdaResNet(LambdaBottleneck, [3,4,6,3], **kwargs)

def lambda_resnet101(**kwargs):
    return LambdaResNet(LambdaBottleneck, [3,4,23,3], **kwargs)

def lambda_resnet152(**kwargs):
    return LambdaResNet(LambdaBottleneck, [3,4,36,3], **kwargs)

def main():
    model_resnet18 = resnet18()
    model_resnet34 = resnet34()
    model_resnet50 = resnet50()
    model_resnet101 = resnet101()
    model_resnet152 = resnet152()

    model_lambda_resnet18 = lambda_resnet18()
    model_lambda_resnet34 = lambda_resnet34()
    model_lambda_resnet38 = lambda_resnet38()
    model_lambda_resnet50 = lambda_resnet50()
    model_lambda_resnet101 = lambda_resnet101()
    model_lambda_resnet152 = lambda_resnet152()
    

    # summary(model_resnet18, input_size=(32, 3, 224, 224))
    # summary(model_resnet50, input_size=(32, 3, 224, 224))
    summary(model_lambda_resnet18, input_size=(32, 3, 224, 224))
    # summary(model_lambda_resnet50, input_size=(32, 3, 224, 224))

if __name__ == "__main__":
    main()