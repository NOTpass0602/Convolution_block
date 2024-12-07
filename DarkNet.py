import torch
import torch.nn as nn
import torch.nn.functional as F


# 残差块（Residual Block）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut连接，确保输入和输出的通道一致
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # 加入残差连接
        return F.relu(out)


# Darknet-53 网络架构
class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()

        # 输入层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # 网络的各个block
        self.layer1 = self._make_layer(32, 64, 1, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 8, 2)
        self.layer4 = self._make_layer(256, 512, 8, 2)
        self.layer5 = self._make_layer(512, 1024, 4, 2)

        # 最后的全连接层（用于分类等任务）
        self.fc = nn.Linear(1024, 1000)  # 假设我们做分类任务（1000类）

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 输入层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # 全局平均池化
        x = x.view(x.size(0), -1)  # 拉伸成一维
        x = self.fc(x)  # 分类输出
        return x


# 测试代码
if __name__ == "__main__":
    model = Darknet53()
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print("Output shape:", output.shape)
