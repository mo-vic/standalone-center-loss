import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        o1 = self.relu1(self.bn1(x))
        o2 = self.conv1(o1)
        o2 = self.relu2(self.bn2(o2))
        z = self.conv2(o2)

        if hasattr(self, "skip_conv"):
            return z + self.skip_conv(o1)
        else:
            return z + x


class ResNet(nn.Module):
    basic_width = [32, 32, 64, 64, 128]

    def __init__(self, input_shape, feat_dim=128, num_classes=10, k=1, n=1):
        super(ResNet, self).__init__()

        self.width = [int(e * k) for e in ResNet.basic_width]

        assert isinstance(input_shape, tuple)
        assert len(input_shape) == 3
        for e in input_shape:
            assert isinstance(e, int)
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, 16, kernel_size=3)
        self.group1 = self._make_group(16, self.width[0], n, stride=1)
        self.group2 = self._make_group(self.width[0], self.width[1], n, stride=2)
        self.group3 = self._make_group(self.width[1], self.width[2], n, stride=1)
        self.group4 = self._make_group(self.width[2], self.width[3], n, stride=2)
        self.group5 = self._make_group(self.width[3], self.width[4], n, stride=1)

        self.bn = nn.BatchNorm2d(self.width[4])
        self.relu = nn.ReLU(inplace=True)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.width[4], feat_dim)
        self.fc2 = nn.Linear(feat_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_group(self, in_channels, out_channels, num_blocks, stride):
        blocks = []
        for idx in range(num_blocks):
            blocks.append(
                BasicBlock(in_channels if idx == 0 else out_channels, out_channels, stride if idx == 0 else 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)

        x = self.relu(self.bn(x))
        x = self.pool3(x)
        x = x.view(-1, self.width[4])
        feature = self.fc1(x)
        x = self.fc2(feature)

        return feature, x


if __name__ == "__main__":
    model = ResNet((3, 32, 32), feat_dim=128, num_classes=10)
    x = torch.rand(2, 3, 32, 32)
    y = model(x)
