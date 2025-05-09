# -*- codeing = utf-8 -*-
# @Time : 2023/6/17 19:27
# @Author : 张庭恺
# @File : 空间和通道自注意力机制.py
# @Software : PyCharm

import torch
from torch import nn
from .ECA_net import ECA_net
from .CA_attention import CA_Atten

# 通道注意力机制
# 里面的ratio：缩减系数是为了减少注意力特征层的参数量
class channels_atten_SE(nn.Module):
	def __init__(self,input_features,ratio = 16):
		super(channels_atten_SE, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.max_pool = nn.AdaptiveMaxPool2d(1)
		self.fc_ = nn.Sequential(nn.Linear(input_features,input_features//ratio,bias=False),
		                         nn.ReLU(),
		                         nn.Linear(input_features//ratio,input_features,bias=False),
		                         )
		self.sigmoid = nn.Sigmoid()

	def forward(self,x):
		b,c,h,w = x.size()
		avg_x = self.avg_pool(x).view(b,c)
		max_x = self.max_pool(x).view(b,c)

		fc_avg = self.fc_(avg_x)
		fc_max = self.fc_(max_x)

		fc_sum = fc_max + fc_avg
		atten_channels = self.sigmoid(fc_sum).view(b,c,1,-1)

		out = atten_channels * x
		return out

'''
通道注意力机制
'''

# 空间注意力机制
class spacial_attention(nn.Module):
	def __init__(self,kernel_size = 7):
		super(spacial_attention, self).__init__()
		# 卷积不改变特征图大小
		self.conv = nn.Conv2d(2,1,kernel_size=kernel_size,stride=1,padding=kernel_size//2)
		self.sigmoid = nn.Sigmoid()

	def forward(self,x):
		b,c,h,w = x.size()
		avg_x = torch.mean(x,dim=1,keepdim=True)
		max_x,_ = torch.max(x,dim=1,keepdim=True)

		avg_max_x = torch.cat([avg_x,max_x],dim=1)

		out = self.sigmoid(self.conv(avg_max_x)) * x



		return out




class CBAM_net(nn.Module):
	def __init__(self,in_features,ratio,kernel_size):
		super(CBAM_net, self).__init__()



		# self.channels_atten_se = channels_atten_SE(in_features,ratio)
		self.channels_atten_eca = ECA_net(in_features,ratio)

		# self.spacial_atten_old = spacial_attention(kernel_size)
		self.spacial_atten_ca = CA_Atten(in_features)

	def forward(self,x,flag = 1):
		b,c,h,w = x.size()
		out = x
		if flag == 0:
			# 先空间注意力然后通道注意力
			spacial_att = self.spacial_atten_old(x)
			out = self.channels_atten_eca(spacial_att)

		# 先通道注意力然后空间注意力
		if flag == 1 :
			channels_att = self.channels_atten_eca(x)

			# out = self.spacial_atten_old(channels_att)
			out = self.spacial_atten_ca(channels_att)

		return out

# model = CBAM_net(64,16,5)
# print(model)
# pseudo_img = torch.ones(4,64,32,32)
#
# out = model(pseudo_img)
#
# print(out)
# print(out.shape)