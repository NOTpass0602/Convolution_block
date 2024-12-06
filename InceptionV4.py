import torch
import torch.nn as nn


# Inception-ResNet Block: 用于组合 Inception 模块和残差连接
class InceptionResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionResNetBlock, self).__init__()

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.residual(x)  # 残差连接
        out = self.conv1x1(x)
        out = self.conv3x3(out)
        out += identity  # 加入残差
        out = self.relu(out)
        return out


# Inception Block with multiple parallel paths
class InceptionBlock(nn.Module):
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()

        # 1x1 卷积
        self.conv1x1 = nn.Conv2d(in_channels, 64, kernel_size=1)

        # 1x1 卷积 + 3x3 卷积
        self.conv1x1_3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1)
        )

        # 1x1 卷积 + 5x5 卷积
        self.conv1x1_5x5 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.Conv2d(32, 128, kernel_size=5, padding=2)
        )

        # 最大池化 + 1x1 卷积
        self.pool1x1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, 32, kernel_size=1)
        )

    def forward(self, x):
        path1 = self.conv1x1(x)
        path2 = self.conv1x1_3x3(x)
        path3 = self.conv1x1_5x5(x)
        path4 = self.pool1x1(x)

        # 拼接各路径的输出
        out = torch.cat([path1, path2, path3, path4], dim=1)
        return out


# Inception v4 模型中对于Inception Block和Inception-ResNet Block的用法
class InceptionV4(nn.Module):
    def __init__(self, num_classes=1000):
        super(InceptionV4, self).__init__()

        # Stem 网络
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # 第1个 Inception 模块
        self.inception_block1 = InceptionBlock(64)

        # 第2个 Inception 模块
        self.inception_block2 = InceptionBlock(256)

        # Inception-ResNet 块
        self.inception_resnet = InceptionResNetBlock(256, 512)

        # 全连接层
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.inception_resnet(x)
        x = x.mean([2, 3])  # 全局平均池化
        x = self.fc(x)
        return x


# 创建并测试 Inception v4 网络
if __name__ == "__main__":
    model = InceptionV4(num_classes=1000)
    print(model)
    input_tensor = torch.randn(1, 3, 224, 224)  # 示例输入大小 [batch_size, channels, height, width]
    output = model(input_tensor)
    print(output.shape)  # 输出预测结果