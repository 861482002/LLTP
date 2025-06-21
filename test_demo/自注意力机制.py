# -*- codeing = utf-8 -*-
# @Time : 2023/6/10 10:57
# @Author : 张庭恺
# @File : 自注意力机制.py
# @Software : PyCharm

import torch
import torch.nn as nn
import numpy as np
import time


class SelfAttention(nn.Module):
	def __init__(self, input_dim):
		super(SelfAttention, self).__init__()
		# 查询矩阵
		self.query = nn.Linear(input_dim, input_dim)  # 4*4096*4
		# 健矩阵
		self.key = nn.Linear(input_dim, input_dim)  # 4*4096*4
		# 值矩阵
		self.value = nn.Linear(input_dim, input_dim)  # 4*4096*4
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x):
		# batch_size, seq_len, _ = x.size()
		# with torch.no_grad():
		query = self.query(x)
		key = self.key(x)
		value = self.value(x)
		# QK相乘 后面除以一个数是为了避免极端情况，有利于softmax的反向传播
		# 进行缩放 scale
		# scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(x.size(-1)).float())
		scores = (torch.bmm(query, key.transpose(1, 2))**(1/10))

		attention_weights = self.softmax(scores)

		weighted_values = torch.bmm(attention_weights, value)
		output = weighted_values

		return output


# points = np.fromfile('000000.bin',dtype=np.float32).reshape(-1,4)
# points = torch.rand((100,4))
time1 = time.perf_counter()
curent_time = time.ctime(time1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# points_tensor = torch.from_numpy(points).to(device)
# points_tensor = torch.unsqueeze(points_tensor,0)
points_tensor = torch.rand((4, 4096, 4), dtype=torch.float).to(device)
input_dim = points_tensor.size()[2]
attention = SelfAttention(input_dim)
attention.to(device)
output = attention(points_tensor)

time2 = time.perf_counter()
cust = (time2 - time1)
print(cust)

print(output)
