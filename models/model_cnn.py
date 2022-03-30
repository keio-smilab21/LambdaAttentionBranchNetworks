import torch
import torch.nn as nn
from torchinfo import summary


class model_CNN_3(nn.Module):
    """
    3層のCNNモデル
    """
    def __init__(self, num_channels: int, num_classes: int=1000):
        super(model_CNN_3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=16, kernel_size=3, 
                                stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, 
                                stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                                stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.avgpool = nn.AdaptiveMaxPool2d((1,1))

        self.fc = nn.Linear(32, num_classes)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x) 

        x = self.conv3(x)
        x =  self.relu(x) 
        x = self.avgpool(x) # 32, 32, 32, 32
        print(x.size()) # 32, 32, 1, 1

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def main():
    model = model_CNN_3(num_channels=3, num_classes=2)
    summary(model, input_size=(32, 3, 512, 512))

if __name__=="__main__":
    main()
