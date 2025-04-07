import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import SimpleNN

# 1. 加载数据集并预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量，并自动归一化到 [0,1]
])

# 下载并加载训练集和测试集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 创建 DataLoader（分批加载数据）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = SimpleNN()

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 4. 训练模型
num_epochs = 10
for epoch in tqdm(range(num_epochs)):
    for images, labels in train_loader:

        # 前向传播 f(x)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播与优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 反向计算梯度
        optimizer.step()       # 更新参数
        torch.save(model.state_dict(), './model.pth')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 评估模型
model.eval()  # 切换到评估模式
correct = 0
total = 0
with torch.no_grad():  # 不计算梯度
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 取概率最大的类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total:.2f}%')