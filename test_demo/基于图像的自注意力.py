# -*- codeing = utf-8 -*-
# @Time : 2023/6/17 16:25
# @Author : 张庭恺
# @File : 基于图像的自注意力.py
# @Software : PyCharm


import torch
import torch.nn as nn


class ImageSelfAttention(nn.Module):
	def __init__(self, in_channels, num_heads):
		super(ImageSelfAttention, self).__init__()

		self.num_heads = num_heads

		self.query_conv = nn.Conv2d(in_channels, in_channels // num_heads, kernel_size=1)
		self.key_conv = nn.Conv2d(in_channels, in_channels // num_heads, kernel_size=1)
		self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

		self.attention = nn.MultiheadAttention(in_channels, num_heads)

		self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		# 尺寸变换
		query = self.query_conv(x)
		key = self.key_conv(x)
		value = self.value_conv(x)

		# 重塑形状
		n, c, h, w = query.size()
		query = query.view(n, self.num_heads, -1, h * w).permute(0, 2, 1, 3)
		key = key.view(n, self.num_heads, -1, h * w).permute(0, 2, 1, 3)
		value = value.view(n, self.num_heads, -1, h * w).permute(0, 2, 1, 3)

		# 注意力计算
		attention_output, _ = self.attention(query, key, value)

		# 重塑形状
		attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
		attention_output = attention_output.view(n, -1, h, w)

		# 输出转换
		output = self.out_conv(attention_output)

		# ReLU激活函数
		output = self.relu(output)

		return output

# 示例用法
in_channels = 64
num_heads = 4

# 创建自注意力模块实例
self_attention = ImageSelfAttention(in_channels, num_heads)

# 定义输入图像
batch_size = 1
input_image = torch.randn(batch_size, in_channels, 32, 32)

# 前向传播
output = self_attention(input_image)
print(output.shape)
