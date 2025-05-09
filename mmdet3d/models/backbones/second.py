# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.runner import BaseModule
from torch import nn as nn

from ..builder import BACKBONES
from ..CBAM_net import *


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

		in_filters = [in_channels, *out_channels[:-1]]
		# note that when stride > 1, conv2d with same padding isn't
		# equal to pad-conv2d. we should use pad-conv2d.
		blocks = []
		for i, layer_num in enumerate(layer_nums):  # [3, 5, 5]
			block = [
				build_conv_layer(
					conv_cfg,
					in_filters[i],
					out_channels[i],
					3,
					stride=layer_strides[i],  # [2, 2, 2]
					padding=1),
				build_norm_layer(norm_cfg, out_channels[i])[1],
				nn.ReLU(inplace=True),
			]
			for j in range(layer_num):  # 一般都是 卷积 标准化 激活
				block.append(
					build_conv_layer(
						conv_cfg,
						out_channels[i],
						out_channels[i],
						3,
						padding=1))
				block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
				block.append(nn.ReLU(inplace=True))

			block = nn.Sequential(*block)
			blocks.append(block)

		# 最后会形成3 个block 每个block又是一个sequential ， 每个sequential的第一层都是一个步长为2的卷积，后面的步长为1
		# 三个block分别 为 4，6，6 的长度
		self.blocks = nn.ModuleList(blocks)

		assert not (init_cfg and pretrained), \
			'init_cfg and pretrained cannot be setting at the same time'
		if isinstance(pretrained, str):
			warnings.warn('DeprecationWarning: pretrained is a deprecated, '
			              'please use "init_cfg" instead')
			self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
		else:
			self.init_cfg = dict(type='Kaiming', layer='Conv2d')

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
		# atten_outs = []
		# for i, res_out in enumerate(outs):
		# 	res_out = self.pseudo_image_att[i](res_out)
		# 	atten_outs.append(res_out)
		# # 在此处增加基于伪图像的自注意力机制
		# return tuple(atten_outs)
		return tuple(outs)
