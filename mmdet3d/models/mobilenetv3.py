# -*- codeing = utf-8 -*-
# @Time : 2023-09-05 14:51
# @Author : 张庭恺
# @File : mobilenetv3.py
# @Software : PyCharm


from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial


'''
调整输入通道到离其最近的一个8的整数倍
'''


def _make_divisiable(ch, divisor=8, min_ch=None):
	if min_ch is None:
		min_ch = divisor
	new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_ch < 0.9 * ch:
		new_ch += divisor
	return new_ch

	pass


class ConvBNActivation(nn.Sequential):
	def __init__(self,
	             in_planes: int,
	             out_planes: int,
	             kernel_size: int = 3,
	             stride: int = 1,
	             groups: int = 1,
	             norm_layer: Optional[Callable[..., nn.Module]] = None,
	             activation_layer: Optional[Callable[..., nn.Module]] = None,
	             ):
		padding = (kernel_size - 1) // 2
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if activation_layer is None:
			activation_layer = nn.ReLU6

		super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
		                                                 out_channels=out_planes,
		                                                 kernel_size=kernel_size,
		                                                 stride=stride,
		                                                 padding=padding,
		                                                 groups=groups,
		                                                 bias=False),
		                                       norm_layer(out_planes),
		                                       activation_layer(inplace=True)
		                                       )


class SEnet(nn.Module):
	def __init__(self, int_c: int, sequeeze_factor: int = 4):
		super(SEnet,self).__init__()
		se_c = _make_divisiable(int_c // sequeeze_factor, 8)
		self.c_fc1 = nn.Conv2d(in_channels=int_c, out_channels=se_c, kernel_size=1)
		self.c_fc2 = nn.Conv2d(in_channels=se_c, out_channels=int_c, kernel_size=1)
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

	def forward(self, x: Tensor) -> Tensor:
		scale = self.avg_pool(x)
		scale = self.c_fc1(scale)
		scale = F.relu6(scale, inplace=True)
		scale = self.c_fc2(scale)
		scale = F.hardsigmoid(scale, inplace=True)
		return x * scale


class InvertedResidualConfig:
	def __init__(self,
	             input_c: int,
	             kernel: int,
	             expanned_c: int,
	             out_c: int,
	             use_se: bool,
	             activation: str,
	             stride: int,
	             width_multi: float
	             ):
		self.input_c = self.adjust_channels(input_c, width_multi)
		self.kernel = kernel
		self.expanned_c = self.adjust_channels(expanned_c, width_multi)
		self.out_c = self.adjust_channels(out_c, width_multi)
		self.use_se = use_se
		self.use_hs = activation == 'HS'
		self.stride = stride

	@staticmethod
	def adjust_channels(channels: int, width_multi: float):
		return _make_divisiable(channels * width_multi, 8)


# Mobilev3里面的   bneck模块的定义
class InvertedResidual(nn.Module):
	def __init__(self,
	             cnf: InvertedResidualConfig,
	             norm_layer: Callable[..., nn.Module]
	             ):
		super(InvertedResidual,self).__init__()

		if cnf.stride not in [1, 2]:
			raise ValueError('不合法的步长')

		self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

		layers: List[nn.Module] = []
		activation_layer = nn.Hardswish if cnf.use_hs else nn.LeakyReLU

		# expand
		if cnf.expanned_c != cnf.input_c:
			layers.append(ConvBNActivation(cnf.input_c,
			                               cnf.expanned_c,
			                               kernel_size=1,
			                               stride=1,
			                               groups=1,
			                               norm_layer=norm_layer,
			                               activation_layer=activation_layer
			                               ))
		# depthwise
		layers.append(ConvBNActivation(cnf.expanned_c,
		                               cnf.expanned_c,
		                               kernel_size=cnf.stride,
		                               # kernel_size=cnf.kernel,
		                               stride=cnf.stride,
		                               groups=cnf.expanned_c,
		                               norm_layer=norm_layer,
		                               activation_layer=activation_layer
		                               ))

		if cnf.use_se:
			layers.append(SEnet(int_c=cnf.expanned_c,
			                    sequeeze_factor=4))

		# pointwiase
		layers.append(ConvBNActivation(in_planes=cnf.expanned_c,
		                               out_planes=cnf.out_c,
		                               kernel_size=1,
		                               stride=1,
		                               groups=1,
		                               norm_layer=norm_layer,
		                               activation_layer=nn.Identity))
		self.bneck = nn.Sequential(*layers)
		self.out_channels = cnf.out_c
		self.stride = cnf.stride

	def forward(self, x: Tensor) -> Tensor:
		out = self.bneck(x)
		if self.use_res_connect:
			out += x
		return out


class Mobilev3(nn.Module):

	def __init__(self,
	             inverted_residual_setting: List[InvertedResidualConfig],
	             last_channels: int,
	             num_classes: int = 1000,
	             block: Optional[Callable[..., nn.Module]] = None,
	             norm_layer: Optional[Callable[..., nn.Module]] = None
	             ):
		super(Mobilev3,self).__init__()

		if inverted_residual_setting is None:
			raise ValueError('配置项不能为空')
		elif not (isinstance(inverted_residual_setting, List) and
		      all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
			raise TypeError('配置项配型错误')

		if block is None:
			block = InvertedResidual

		if norm_layer is None:
			# 为bn2d传入两个默认参数 ，之后就不用每次都传入默认参数
			norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

		layers: List[nn.Module] = []

		# 构建bneck结构
		firstconv_out_c = inverted_residual_setting[0].input_c
		layers.append(ConvBNActivation(3,
		                               firstconv_out_c,
		                               kernel_size=3,
		                               stride=2,
		                               norm_layer=norm_layer,
		                               activation_layer=nn.Hardswish))

		for cnf in inverted_residual_setting:
			layers.append(block(cnf, norm_layer=norm_layer))

		# 构建网络最后几层的结构
		lastconv_input_c = inverted_residual_setting[-1].out_c
		lastconv_out_c = lastconv_input_c * 6
		layers.append(ConvBNActivation(in_planes=lastconv_input_c,
		                               out_planes=lastconv_out_c,
		                               kernel_size=1,
		                               norm_layer=norm_layer,
		                               activation_layer=nn.Hardswish))

		self.features = nn.Sequential(*layers)
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Sequential(nn.Linear(lastconv_out_c, last_channels),
		                                nn.Hardswish(inplace=True),
		                                nn.Dropout(p=0.2, inplace=True),
		                                nn.Linear(last_channels, num_classes))

		# 初始化参数
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

	def _forward_impl(self, x: Tensor) -> Tensor:
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, start_dim=1)
		x = self.classifier(x)

		return x
	def forward(self,x):
		return self._forward_impl(x)

def mobilev3_large(class_num=1000) -> nn.Module:
	width_multi = 1.0
	bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
	bneck_cnf_make_divisiable = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

	reduce_divider = 1
	cnf_list = [
		bneck_conf(16, 3, 16, 16, False, "RE", 1),
		bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1
		bneck_conf(24, 3, 72, 24, False, "RE", 1),
		bneck_conf(24, 5, 72, 40, True, "RE", 2),  # C2
		bneck_conf(40, 5, 120, 40, True, "RE", 1),
		bneck_conf(40, 5, 120, 40, True, "RE", 1),
		bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3
		bneck_conf(80, 3, 200, 80, False, "HS", 1),
		bneck_conf(80, 3, 184, 80, False, "HS", 1),
		bneck_conf(80, 3, 184, 80, False, "HS", 1),
		bneck_conf(80, 3, 480, 112, True, "HS", 1),
		bneck_conf(112, 3, 672, 112, True, "HS", 1),
		bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4
		bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
		bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
	]
	last_c = 1280
	class_num = class_num

	return Mobilev3(cnf_list,
	                last_c,
	                class_num)


if __name__ == '__main__':
	mobilenetv3 = mobilev3_large(1000)

	x = torch.rand((3,3,224,224))
	y = mobilenetv3(x)
	print(mobilenetv3)
	print(y.shape)
