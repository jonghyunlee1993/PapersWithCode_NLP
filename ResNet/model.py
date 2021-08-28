import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.activation = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layers(block, inplane=64, outplane=64, num_blocks=num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, inplane=64, outplane=128, num_blocks=num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, inplane=64, outplane=256, num_blocks=num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, inplane=64, outplane=512, num_blocks=num_blocks[3], stride=2)
        
        self.average_pool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layers(self, block, inplane, outplane, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            layers.append(block(inplane, outplane, stride))
            self.in_channels = outplane

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.average_pool(x)
        x = x.view(x.size(0), -1) # flatten
        x = self.fc(x)
        
        return x
    
    
class BasicBlock(nn.Module):
    def __init__(self, inplane, outplane, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(outplane)
        self.conv2 = nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(outplane)
        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x += residual
        x = self.activation(x)
        
        return x
        
        
# class BottleNeckBlock(nn.Module):
#     def __init__(self, inplane, outplane):
#         super().__init__()
#         self.conv1 = nn.Conv2d(inplane, outplane, 1)
#         self.conv2 = nn.Conv2d(inplane, outplane, 3)
#         self.conv3 = nn.Conv2d(inplane, outplane * 4, 1)
#         self.activation = nn.ReLU(inplace=True)
        
#     def forward(self, x):
#         residual = x
        
#         x = self.conv1(x)
#         x = self.activation(x)
#         x = self.conv2(x)
#         x = self.activation(x)
#         x = self.conv3(x)
#         x = self.activation(x)
        
#         x += residual 
#         x = self.activation(x)
        
#         return x


if __name__ == "__main__":
    input_shape = 3, 256, 256
    sample_input = torch.tensor((input_shape))
    
    model_ResNet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    
    print(model_ResNet18)