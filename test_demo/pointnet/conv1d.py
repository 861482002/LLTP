# -*- codeing = utf-8 -*-
# @Time : 2023/6/10 19:48
# @Author : 张庭恺
# @File : conv1d.py
# @Software : PyCharm

import torch
import torch.nn as nn

# 定义一维卷积层
in_channels = 1  # 输入通道数
out_channels = 16  # 输出通道数
kernel_size = 3  # 卷积核大小
stride = 1  # 步长
padding = 1  # 填充

conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

# 前向传播计算
batch_size = 5
input_length =100
x = torch.randn(batch_size, in_channels, input_length)
output = conv1d(x)
print(output.size())



# import torch
# import torch.nn as nn
#
#
# # 定义一个简单的文本分类模型
# class TextClassifier(nn.Module):
# 	def __init__(self, vocab_size, embedding_dim, num_classes):
# 		super(TextClassifier, self).__init__()
# 		self.embedding = nn.Embedding(vocab_size, embedding_dim)
# 		self.conv1d = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
# 		self.relu = nn.ReLU()
# 		self.fc = nn.Linear(128, num_classes)
#
# 	def forward(self, x):
# 		embedded = self.embedding(x)
# 		embedded = embedded.permute(0, 2, 1)  # 调整维度顺序以适应一维卷积层
# 		conv_out = self.conv1d(embedded)
# 		conv_out = self.relu(conv_out)
# 		pooled = torch.max(conv_out, dim=2)[0]
# 		logits = self.fc(pooled)
# 		return logits
#
#
# # 创建模型实例
# vocab_size = 10000
# embedding_dim = 100
# num_classes = 2
#
# model = TextClassifier(vocab_size, embedding_dim, num_classes)
#
# # 定义输入数据
# batch_size = 32
# seq_length = 100
#
# x = torch.randint(0, vocab_size, (batch_size, seq_length))
#
# # 前向传播计算
# logits = model(x)
#
# print(logits.shape)  # 输出结果的形状


