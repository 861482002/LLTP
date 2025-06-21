# -*- codeing = utf-8 -*-
# @Time : 2023/6/14 11:01
# @Author : 张庭恺
# @File : 图片transformer.py
# @Software : PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
	def __init__(self, in_channels):
		super(SelfAttention, self).__init__()

		self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, x):
		batch_size, channels, height, width = x.size()

		query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
		key = self.key_conv(x).view(batch_size, -1, width * height)
		value = self.value_conv(x).view(batch_size, -1, width * height)

		attention_weights = F.softmax(torch.bmm(query, key), dim=2)
		attention_output = torch.bmm(value, attention_weights.permute(0, 2, 1))
		attention_output = attention_output.view(batch_size, channels, height, width)

		out = self.gamma * attention_output + x
		return out


# 定义模型
class ImageTransformer(nn.Module):
	def __init__(self, in_channels, hidden_dim, num_heads, num_layers):
		super(ImageTransformer, self).__init__()

		self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
		self.attentions = nn.ModuleList([
			SelfAttention(hidden_dim) for _ in range(num_layers)
		])
		self.fc = nn.Linear(hidden_dim, in_channels)

	def forward(self, x):
		x = self.conv(x)

		for attention in self.attentions:
			x = attention(x)

		x = torch.mean(x, dim=(2, 3))  # 对特征图进行全局平均池化
		x = self.fc(x)

		return x


# 创建模型实例
in_channels = 3  # 输入图像通道数
hidden_dim = 64  # 隐藏层维度
num_heads = 4  # 注意力头数
num_layers = 2  # 注意力层数

model = ImageTransformer(in_channels, hidden_dim, num_heads, num_layers)

# 将输入图像传递给模型
input_image = torch.randn(1, in_channels, 32, 32)  # 输入图像的大小为32x32
output = model(input_image)
print(output.shape)  # 输出的形状
