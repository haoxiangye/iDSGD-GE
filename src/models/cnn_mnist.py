import torch
import torch.nn.functional as F
from torch import nn

from src.models import BaseModel


class Model(BaseModel):
    def __init__(self, channels=32, num_classes=10, **kwargs):
        super(Model, self).__init__()
        # 第一层卷积层，使用32个3x3的卷积核，输出通道数为32
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # 第二层卷积层，使用64个3x3的卷积核，输出通道数为64
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 池化层，使用2x2的池化核，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层，将池化后的特征图展平，然后连接到128个神经元
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # 输出层，将全连接层的输出连接到10个神经元，对应10个类别
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # 通过第一层卷积层和激活函数
        x = self.pool(F.relu(self.conv1(x)))
        # 通过第二层卷积层和激活函数
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图，准备输入到全连接层
        x = x.view(-1, 64 * 7 * 7)
        # 通过第一个全连接层和激活函数
        x = F.relu(self.fc1(x))
        # 通过输出层，得到最终的分类结果
        x = self.fc2(x)
        return x


# class Model(BaseModel):
#     def __init__(self, channels=32, num_classes=10, **kwargs):
#         super(Model, self).__init__()
#         # 第一层卷积层，使用32个3x3的卷积核，输出通道数为32
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
#         # 第二层卷积层，使用32个3x3的卷积核，输出通道数为32
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
#         # 池化层，使用2x2的池化核，步长为2
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         # 第三层卷积层，使用64个3x3的卷积核，输出通道数为64
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         # 第四层卷积层，使用64个3x3的卷积核，输出通道数为64
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
#         # 全连接层，将池化后的特征图展平，然后连接到128个神经元
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         # 输出层，将全连接层的输出连接到10个神经元，对应10个类别
#         self.fc2 = nn.Linear(128, num_classes)

#     def forward(self, x):
#         # 通过第一层卷积层和激活函数
#         x = self.pool(F.relu(self.conv1(x)))
#         # 通过第二层卷积层和激活函数
#         x = self.pool(F.relu(self.conv2(x)))
#         # 通过第三层卷积层和激活函数
#         x = self.pool(F.relu(self.conv3(x)))
#         # 通过第四层卷积层和激活函数
#         x = self.pool(F.relu(self.conv4(x)))
#         # 展平特征图，准备输入到全连接层
#         x = x.view(-1, 64 * 7 * 7)
#         # 通过第一个全连接层和激活函数
#         x = F.relu(self.fc1(x))
#         # 通过输出层，得到最终的分类结果
#         x = self.fc2(x)
#         return x


