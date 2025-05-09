# -*- codeing = utf-8 -*-
# @Time : 2023/6/17 18:30
# @Author : 张庭恺
# @File : pseudo_image_transformer.py
# @Software : PyCharm

import torch
from torch import nn
import torchvision
from timm import models
resnet = torchvision.models.resnet18()


class SE_net(nn.Module):
	def __init__(self, in_channels, ratio=16):
		super(SE_net, self).__init__()
		# 全局的平均池化
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.sqe_fc = nn.Sequential(
			# 第一次全连接的神经元较少
			nn.Linear(in_channels, in_channels // ratio, bias=False),
			# 这个激活函数不需要使用sigmoid，用更好的非线性激活
			nn.ReLU(),
			nn.Linear(in_channels // ratio, in_channels, bias=False),
			# 这个激活函数需要将
			nn.Sigmoid()
		)

	def forward(self, x):
		b, c, h, w = x.size()
		# 输出  b，c，1，1
		avg_x = self.avg_pool(x).view(b, c)
		fc_x = self.sqe_fc(avg_x)
		# print(fc_x)
		channels_atten_weight = fc_x.view(b, c, 1, -1)
		# 使用到广播机制
		out = x * channels_atten_weight
		return out


# pseudo_img = torch.ones((3, 64, 256, 256))
# model = SE_net(64, 16)
# print(model)
# out = model(pseudo_img)
# print(out.shape)
models.SwinTransformer