"""③ 小 ResNet 风格 CNN，用于对转图结果做二分类。"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class SignalClassifier(nn.Module):
    """输入 (B, C, H, W)，输出 1 维 logit（人工=0 / 机器=1）。"""

    def __init__(self, in_channels: int = 1, base_width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 7, 2, 3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(base_width, base_width, 2, stride=1)
        self.layer2 = self._make_layer(base_width, base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_width * 2, base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_width * 4, base_width * 8, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_width * 8, 1)

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [BasicBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return self.fc(x.flatten(1))


if __name__ == "__main__":
    model = SignalClassifier(in_channels=1)
    dummy = torch.randn(2, 1, 512, 512)
    print("输出形状:", model(dummy).shape)   # 期望 [2, 1]