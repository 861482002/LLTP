# -*- codeing = utf-8 -*-
# @Time : 2023/6/9 20:02
# @Author : 张庭恺
# @File : SVM.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.optim as optim

# 准备数据集（示例数据）
X = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=torch.float32)
y = torch.tensor([0, 0, 1, 1], dtype=torch.float32)

# 定义线性 SVM 模型
class SVM(nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入特征维度为2，输出维度为1

    def forward(self, x):
        return self.linear(x)

# 创建 SVM 模型实例
model = SVM()

# 定义损失函数和优化器
criterion = nn.HingeEmbeddingLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 在训练集上进行训练
num_epochs = 5
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs.squeeze(), y)
    loss.backward()
    optimizer.step()

# 在测试集上进行预测
with torch.no_grad():
    predicted = model(X).squeeze().sign()
    accuracy = (predicted == y).float().mean()

print("准确率:", accuracy.item())

