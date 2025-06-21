# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn
from typing import Callable, List, Optional
from functools import partial
from ..builder import BACKBONES
from ..CBAM_net import *
from ..mobilenetv3 import *

@BACKBONES.register_module()
class SECOND(BaseModule):
	"""Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """




	'''
	backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),'''
	def __init__(self,
	             in_channels=128,
	             out_channels=[128, 128, 256],
	             layer_nums=[3, 5, 5],
	             layer_strides=[2, 2, 2],
	             norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
	             conv_cfg=dict(type='Conv2d', bias=False),
	             init_cfg=None,
	             pretrained=None):
		super(SECOND, self).__init__(init_cfg=init_cfg)
		assert len(layer_strides) == len(layer_nums)
		assert len(out_channels) == len(layer_nums)

		'''
                x:list[3]
        		x[0]: 4,64  ,248    ,216
        		x[1]: 4,128 ,124    ,108
        		x[2]: 4,256 ,62     ,54

        '''
		# feature_channels = [64, 128, 256]
		# ratios = [4, 16, 32]
		# kernel_size = [7, 5, 3]
		# self.pseudo_image_att = []
		# device = torch.device('cuda')
		# for i, params in enumerate(feature_channels):
		# 	cbam_net = CBAM_net(feature_channels[i], ratios[i], kernel_size[i]).to(device)
		# 	self.pseudo_image_att.append(cbam_net)
		# self.pse_img_att = nn.ModuleList(self.pseudo_image_att)

		in_filters = [in_channels, *out_channels[:-1]]      #就是定义三层卷积的输入参数 每一层的输入是上一层的输出 最后一层的输出参数用不到
		# note that when stride > 1, conv2d with same padding isn't
		# equal to pad-conv2d. we should use pad-conv2d.
		blocks = []
		norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
		leakyrelu = nn.LeakyReLU

		width_multi = 1.0
		bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
		conf_settings_stride2 = [
			bneck_conf(64,3,64,64,True,'RE',2),
			bneck_conf(64,3,128,128,True,'RE',2),
			bneck_conf(128,3,128,256,True,'RE',2),

		]

		for i ,layer_num in enumerate(layer_nums):
			bneck = []
			bneck.append(InvertedResidual(conf_settings_stride2[i],norm_layer))

			for j in range(layer_num):
				# depthwise 卷积
				bneck.append(ConvBNActivation(in_planes=conf_settings_stride2[i].out_c,
				                              out_planes=conf_settings_stride2[i].out_c,
				                              kernel_size=3,
				                              stride=1,
				                              groups=conf_settings_stride2[i].out_c,
				                              norm_layer=norm_layer,
				                              activation_layer=leakyrelu))
				# pointwise 卷积
				bneck.append(ConvBNActivation(in_planes=conf_settings_stride2[i].out_c,
				                              out_planes=conf_settings_stride2[i].out_c,
				                              kernel_size=1,
				                              stride=1,
				                              groups=1,
				                              norm_layer=norm_layer,
				                              activation_layer=leakyrelu))

			bneck = nn.Sequential(*bneck)
			blocks.append(bneck)
		self.blocks = nn.ModuleList(blocks)




		# for i, layer_num in enumerate(layer_nums):  # [3, 5, 5]
		# 	block = [
		# 		build_conv_layer(
		# 			conv_cfg,
		# 			in_filters[i],
		# 			out_channels[i],
		# 			3,
		# 			stride=layer_strides[i],  # [2, 2, 2]
		# 			padding=1),
		# 		build_norm_layer(norm_cfg, out_channels[i])[1],
		# 		nn.ReLU(inplace=True),
		# 	]
		# 	for j in range(layer_num):  # 一般都是 卷积 标准化 激活
		# 		block.append(
		# 			build_conv_layer(
		# 				conv_cfg,
		# 				out_channels[i],
		# 				out_channels[i],
		# 				3,
		# 				padding=1))
		# 		block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
		# 		block.append(nn.ReLU(inplace=True))
		#
		# 	block = nn.Sequential(*block)
		# 	blocks.append(block)

		# 最后会形成3 个block 每个block又是一个sequential ， 每个sequential的第一层都是一个步长为2的卷积，后面的步长为1
		# 三个block分别 为 4，6，6 的长度
		self.blocks = nn.ModuleList(blocks)

		# assert not (init_cfg and pretrained), \
		# 	'init_cfg and pretrained cannot be setting at the same time'
		# if isinstance(pretrained, str):
		# 	warnings.warn('DeprecationWarning: pretrained is a deprecated, '
		# 	              'please use "init_cfg" instead')
		# 	self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
		# else:
		# 	self.init_cfg = dict(type='Kaiming', layer='Conv2d')
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight)
				nn.init.zeros_(m.bias)

	def forward(self, x):
		"""Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
		outs = []
		for i in range(len(self.blocks)):
			x = self.blocks[i](x)
			outs.append(x)

		'''
        x:list[3]
		x[0]: 4,64  ,248    ,216
		x[1]: 4,128 ,124    ,108
		x[2]: 4,256 ,62     ,54

        '''
		# TODO
		# 在这里每层之间都加一个残差边
		# atten_outs = []
		# for i, res_out in enumerate(outs):
		# 	res_out = self.pseudo_image_att[i](res_out)
		# 	atten_outs.append(res_out)
		# # 在此处增加基于伪图像的自注意力机制
		# return tuple(atten_outs)
		return tuple(outs)
