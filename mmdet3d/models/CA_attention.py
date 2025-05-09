# -*- codeing = utf-8 -*-
# @Time : 2023/6/18 19:22
# @Author : 张庭恺
# @File : CA_attention.py
# @Software : PyCharm

import torch
from torch import nn

class CA_Atten(nn.Module):
	# ratio：缩减系数，为了减少自注意力层的参数量
	def __init__(self,in_channels,ratio = 16):
		super(CA_Atten, self).__init__()
		self.conv1 = nn.Conv2d(in_channels,in_channels//ratio,kernel_size=1,bias=False)
		self.bn = nn.BatchNorm2d(in_channels // ratio)
		self.relu = nn.ReLU()

		# 对高进行卷积特征提取
		self.cov2_1 = nn.Conv2d(in_channels // ratio, in_channels,kernel_size=1)
		# 对宽进行卷积特征提取
		self.cov2_2 = nn.Conv2d(in_channels // ratio, in_channels,kernel_size=1)

		self.sigmoid = nn.Sigmoid()


	def forward(self,x):
		b,c,h,w = x.size()
		# 对宽方向进行特征浓缩
		# b,c,h,w
		# b,c,h,1 --> transpose b,c,1,h
		atten_w = torch.mean(x,3,keepdim=True).transpose(3,2)
		# 对高方向进行特征浓缩
		# b,c,1,w
		atten_h = torch.mean(x,2,keepdim=True)
		# b,c,1,h+w
		# 这里需要看清楚 哪个是 W 哪个是 H
		cat_w_h = torch.cat([atten_w,atten_h],dim=3)
		# b,c/r ,1, h+w
		cat_w_h = self.relu(self.bn(self.conv1(cat_w_h)))

		# 这样写的话 w 和 h 维度就不对应了
		# split_x_h,spilt_x_w = cat_h_w.split([h, w], 3)

		# split_x_w : b,c,1,h
		# spilt_x_h : b,c,1,w
		split_x_w,spilt_x_h = cat_w_h.split([h, w], 3)

		# b,c,h,1
		spilt_x_w_conv_sigmoid = self.sigmoid(self.cov2_1(split_x_w)).transpose(3,2)
		# b,c,1,w
		spilt_x_h_conv_sigmoid = self.sigmoid(self.cov2_2(spilt_x_h))

		# 这是把最后的注意力权重矩阵与原始图像统一大小
		atten_w_weight = spilt_x_w_conv_sigmoid.repeat(1,1,1,w)

		# 这是把最后的注意力权重矩阵与原始图像统一大小
		atten_h_weight = spilt_x_h_conv_sigmoid.repeat(1,1,h,1)

		out = x * atten_h_weight * atten_w_weight
		return out
# model = CA_Atten(64,16)
# print(model)
# pseudo_img = torch.ones(4,64,32,32)
#
# out = model(pseudo_img)
#
# print(out)
# print(out.shape)



