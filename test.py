import torch

import tensorflow as tf

from model import SimpleNN

# 定义一个简单的模型架构，这里以一个简单的全连接网络为例

# 加载权重文件
weight_path = './model.pth'
# 实例化模型
model1 = SimpleNN()
# 加载模型的状态字典
model_state_dict = torch.load(weight_path)
model1.load_state_dict(model_state_dict)
# 设置模型为评估模式

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 模拟验证集，通常是从训练集中划分一部分作为验证集
validation_images = train_images[:5000]
validation_labels = train_labels[:5000]
train_images = train_images[5000:]
train_labels = train_labels[5000:]

print("MNIST数据集的类型是： %s" % (type(mnist)))
print("训练集的数量是：%d" % len(train_images))
print("验证集的数量是：%d" % len(validation_images))
print("测试集的数量是：%d" % len(test_images))

# 这里简单取一个测试样本进行推理
image = torch.tensor(test_images[6666], dtype=torch.float32)
image2 = torch.squeeze(image,dim=0)
import PIL.Image as Image
img = Image.fromarray(image2.numpy(), 'RGB')
# Image做一个resize操作
img = img.resize((280,280))
img.show()
# 增加一个维度以匹配模型输入 (batch_size, channels, height, width)
image = image.unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    output = model1(image)
    output = output.cpu().detach().numpy()
    import numpy as np
    # 获取数组中最大值对应的索引
    maxValue = np.argmax(output)
    print("模型输出：", maxValue)
