# 2. 定义模型
from torch import nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()  # 将 28x28 展平为 784 维
        self.fc1 = nn.Linear(784, 128)  # 全连接层（输入784，输出128）
        self.relu = nn.ReLU()          # 激活函数
        self.fc2 = nn.Linear(128, 10)  # 输出层（输出10类）
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x