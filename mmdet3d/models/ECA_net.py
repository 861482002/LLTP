# -*- codeing = utf-8 -*-
# @Time : 2023/6/18 17:38
# @Author : 张庭恺
# @File : ECA_net.py
# @Software : PyCharm

import torch
from torch import nn
import math

# 通道特征注意力
'''
由于SE_net是通过全连接层来对经过全局平均池和全局最大池来提取特征的，但是ECAnet的作者认为这样可能会带来副作用
使用1d卷积能够更好的提取特征

'''


class ECA_net(nn.Module):
	def __init__(self, channels , gamma = 2 , b = 1):
		super(ECA_net, self).__init__()
		self.kernel_size = int(abs(math.log(channels,2)+b) / gamma)
		self.kernel_size = self.kernel_size if self.kernel_size % 2 else self.kernel_size + 1
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.conv1d = nn.Conv1d(1,1,kernel_size=self.kernel_size,bias=False,padding=self.kernel_size//2)

		self.sigmoid = nn.Sigmoid()

	def forward(self,x):
		b,c,h,w = x.size()                      #2D卷积的 forward输入数据的shape为 ( batchsiaze , channel , h , w )
		avg_x = self.avg_pool(x).view(b,1,c)    #1D卷积的 forward输入数据的shape为 ( batchsiaze, 一个样本的时间步长, 一个步长的特征数)

		atten = self.sigmoid(self.conv1d(avg_x)).view(b,c,1,1)

		out = atten * x
		return out
#
# model = ECA_net(64)
# print(model)
# pseudo_img = torch.randn(4,64,32,32)
#
#
# out = model(pseudo_img)
# print(out.shape)
