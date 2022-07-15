import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, stride=1, groups = 32, width_per_group = 4, downsample=None):
        group_width = groups * width_per_group
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, group_width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, 3, stride, 1, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, group_width*self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(group_width*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        out += shortcut
        out = self.relu(out)

        return out

class ResNeXt(nn.Module):
    in_planes = 64
        
    def __init__(self, block, num_blocks, groups = 32, width_per_group = 4, num_classes=1000):
        super(ResNeXt, self).__init__()

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False) # Note: difference of origin paper, stride change to 1, but this change will add much more calculation
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.do = nn.Dropout(0.5)
        self.fc = nn.Linear(2048, num_classes)
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != self.groups * self.width_per_group * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, self.groups * self.width_per_group * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.groups * self.width_per_group * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, stride, groups=self.groups, width_per_group=self.width_per_group, downsample=downsample))
        self.in_planes = block.expansion * self.groups * self.width_per_group
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, groups=self.groups, width_per_group=self.width_per_group))
        self.width_per_group *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        # x = self.maxpool(x) # Note: difference of origin paper, because input images size is 32 * 32, it is too small for ResNet
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.do(x)
        x = self.fc(x)
        return x

def _resnext(arch, block, layers, pretrained, num_classes, groups, width_per_group, progress=True):
    model = ResNeXt(block, layers, num_classes=num_classes, groups=groups, width_per_group=width_per_group)
    model_dict = model.state_dict()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)

        # remove fc 
        if num_classes != 1000:
            del state_dict['fc.weight']
            del state_dict['fc.bias']
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


def resnext50(num_classes, pretrained=False):
    groups = 32
    width_per_group = 4
    model = _resnext('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], pretrained, num_classes=num_classes, groups=groups, width_per_group=width_per_group)
    return model

def resnext101(num_classes, pretrained=False):
    groups = 32
    width_per_group = 8
    model = _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], pretrained, num_classes=num_classes, groups=groups, width_per_group=width_per_group)
    return model

if __name__ == "__main__":
    net = resnext101(10, True)
    print(net.parameters())

    x = torch.zeros(10, 3, 32, 32)
    out = net(x)
    print(out.shape)