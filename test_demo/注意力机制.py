# -*- codeing = utf-8 -*-
# @Time : 2023/6/10 11:59
# @Author : 张庭恺
# @File : 注意力机制.py
# @Software : PyCharm
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.W = nn.Linear(input_dim, input_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入x的形状：(batch_size, sequence_length, input_dim)

        # 计算注意力权重
        attention_weights = self.softmax(self.W(x))  # 形状：(batch_size, sequence_length, input_dim)

        # 对输入x加权求和
        attended_x = torch.matmul(attention_weights.transpose(1, 2), x)  # 形状：(batch_size, input_dim, sequence_length)

        return attended_x, attention_weights


# 创建一个输入张量进行测试
batch_size = 4
sequence_length = 6
input_dim = 8

x = torch.randn(batch_size, sequence_length, input_dim)

# 创建自注意力层实例
attention = SelfAttention(input_dim)


# 前向传播
attended_x, attention_weights = attention(x)

print("Input shape:", x.shape)
print("Attended output shape:", attended_x.shape)
print("Attention weights shape:", attention_weights.shape)
