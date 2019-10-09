'''ResNet_v2 in PyTorch.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152", "get_P_model", "Bottleneck", "ResNet"]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        pre = F.relu(self.bn0(x))
        out = F.relu(self.bn1(self.conv1(pre)))
        out = self.conv2(out)

        if len(self.shortcut)==0:
            out += self.shortcut(x)
        else:
            out += self.shortcut(pre)
        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['shortcut']
    def __init__(self, in_planes, planes, stride=1, shortcut=None):
        super(Bottleneck, self).__init__()
        self.bn0 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace = True)
        

    def forward(self, x):
        identity = x
        out = self.bn0(x)
        out = self.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out += identity

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.relu = nn.ReLU(inplace = True)
        self.avgpool2d = nn.AdaptiveAvgPool2d((1,1))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.xavier_uniform_(m.bias)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            if stride != 1 or self.in_planes != block.expansion * planes:
                shortcut = nn.Sequential(
                    nn.Conv2d(self.in_planes, block.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.expansion*planes)
                )
            else:
                shortcut = None
            layers.append(block(self.in_planes, planes, stride, shortcut = shortcut))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #print(self.conv1(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool2d(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        #print(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes = num_classes)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


# def test():
#     net = ResNet50()
#     y = net(torch.randn(1,3,32,32))
#     # print(net)
#     # print(y.size())

# test()

if __name__ == '__main__':
    model = ResNet50()
    input = torch.randn((1, 3, 8, 8))
    output = model(input)

