# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
from torch import nn as nn

from ..transformer import *
from mmdet3d.ops import build_sa_module
from ..builder import BACKBONES
from .base_pointnet import BasePointNet


# class SelfAttention_xyz(nn.Module):
# 	def __init__(self, input_dim):
# 		super(SelfAttention_xyz, self).__init__()
# 		# 查询矩阵
# 		self.query = nn.Linear(input_dim, input_dim)
# 		# 键矩阵
# 		self.key = nn.Linear(input_dim, input_dim)
# 		# 值矩阵
# 		self.value = nn.Linear(input_dim, input_dim)
# 		self.sofa_max = nn.Softmax(dim=-1)
#
# 	def forward(self, x):
# 		query = self.query(x)
# 		key = self.key(x)
# 		value = self.value(x)
#
# 		# x : (b,n,features) 2*4096*4
# 		# QK相乘得到相似度得分
# 		# score : shape(b,4096,4096)
# 		score = (torch.bmm(query, torch.transpose(key, 1, 2))) / torch.sqrt(torch.tensor(x.size(-1))).float()
#
# 		attention_weights = self.sofa_max(score)
# 		weighted_valure = torch.bmm(attention_weights, value)
# 		out_put = weighted_valure
#
# 		return out_put
#
#
# #特征注意力
# class SelfAttention_feature(nn.Module):
# 	def __init__(self, input_dim):
# 		super(SelfAttention_feature, self).__init__()
# 		# 查询矩阵
# 		self.query = nn.Linear(input_dim, input_dim)
# 		# 键矩阵
# 		self.key = nn.Linear(input_dim, input_dim)
# 		# 值矩阵
# 		self.value = nn.Linear(input_dim, input_dim)
# 		self.sofa_max = nn.Softmax(dim=-1)
#
# 	def forward(self, x):
# 		'''
# 		输入的feature：1*1*4096
# 		需要先进行transpose
# 		'''     # x : (B, 256 ,512) --> transpose --> (B , 512 , 256)
# 		x = torch.transpose(x, 1, 2)
# 		query = self.query(x)
# 		key = self.key(x)
# 		value = self.value(x)
#
# 		# Q,K,V 都是 ( B , 512 ,256 )
# 		# QK相乘得到相似度得分
# 		# score : shape(b,512,512)
# 		# score = (torch.bmm(query, torch.transpose(key, 1, 2))) / torch.sqrt(torch.tensor(x.size(-1))).float()
# 		score = (torch.bmm(query, torch.transpose(key, 1, 2))) / torch.sqrt(torch.tensor(x.size(-1))).float()
#
# 		attention_weights = self.sofa_max(score)
# 		# weighted_valure: (B , 512 ,256)
# 		weighted_valure = torch.bmm(attention_weights, value)
# 		# 最后再进行一个转置，将特征维度放到第二维
# 		# out_put: (B, 256, 512)
# 		out_put = weighted_valure.transpose(1, 2)
#
# 		return out_put
#
# # 线性变化
# class FeedForward(nn.Module):
# 	'''
# 	w2(relu((w1(layer_norm(x))+b1)))+b2
#
# 	'''
# 	def __init__(self,input_dim,ff_dim,dropout = 0.1):
# 		super(FeedForward, self).__init__()
# 		self.w1 = nn.Linear(in_features=input_dim,out_features=ff_dim)
# 		self.w2 = nn.Linear()
# 		self.lay_norm = nn.LayerNorm(input_dim)
# 		self.relu = nn.ReLU()
# 		self.dropout1 = nn.Dropout(dropout)
# 		self.dropout2 = nn.Dropout(dropout)
#
# 	def forward(self,x):
# 		inter = self.dropout1(self.relu(self.w1((self.lay_norm(x)))))
# 		outer = self.dropout2(self.w2(inter))
# 		return outer
#
# class Generator(nn.Module):
# 	def __init__(self,input_dim):
# 		super(Generator, self).__init__()
# 		self.linear = nn.Linear(input_dim,)
#
# 	def forward(self,x):
# 		return F.log_softmax(self.linear,dim=-1)
#
#
# # ( B , Features , N_P) --> ( 2 , 64 , 512 )
# class MultiHead_SelfAttention(nn.Module):
# 	def __init__(self, input_dim, head_num):
# 		super(MultiHead_SelfAttention, self).__init__()
# 		self.head_num = head_num
# 		self.input_dim = input_dim
# 		assert input_dim % head_num == 0
# 		self.sub_dim = input_dim // head_num
# 		# 先是来自同源的 QKV 然后再做自注意力
# 		self.query = nn.Linear(self.input_dim, self.input_dim)
# 		self.key = nn.Linear(self.input_dim, self.input_dim)
# 		self.value = nn.Linear(self.input_dim, self.input_dim)
#
# 		self.linear = nn.Linear(self.input_dim, self.input_dim)
# 		self.softmax = nn.Softmax(-1)
#
# 	# ( batch_size , Features , num_points) --> ( 2 , 256 , 512 ) --> (2,512,256)
# 	def forward(self, x):  # x: (2,256,512) --> transpose       最后需要把头数放到序列大小前面
# 		Q = self.query(torch.transpose(x, 1, 2)).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1,2)
# 		K = self.key(torch.transpose(x, 1, 2)).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1,2)
# 		V = self.value(torch.transpose(x, 1, 2)).view(x.size(0), -1, self.head_num, self.sub_dim).transpose(1,2)
# 		# head_num = 8
# 		# 输入形状为（B , head_num , 512 , 32 ）
# 		attention = self.linear(self.multihead_self_att(Q, K, V))
# 		# 进行转置于原始形状对应
# 		# （B, 256, 512）
# 		output = attention.transpose(1,2)
#
# 		return output
#
# 	def multihead_self_att(self, Q, K, V):
# 		# 输入形状为（B , head_num , 512 , 32 ）
# 		batch_size, head_num, point_num, sub_features = Q.size()
# 		# weighted_values = []
# 		# (B ,head_num , 512 , 512 )
# 		score = torch.matmul(Q,K.transpose(-1,-2))/ torch.sqrt(torch.tensor(Q.size(-1)))
#
# 		# softmax
# 		atten_weights = F.softmax(score,-1)
# 		# 注意力得分矩阵于V相乘
# 		# （B, head_num, 512, 32 ）
# 		weighted_atten = torch.matmul(atten_weights,V)
# 		# （B, head_num, 512, 32 ） transpose --> （B, 512, head_num, 32 ） --> （B, 512, 256 ）
# 		output = weighted_atten.transpose(1,2).contiguous().vies(batch_size,-1,head_num*sub_features)
#
# 		'''
# 		这里面是没有用matmul的结果
# 		for i in range(head_num):
# 			# 任何一个sub_Q,K,V的形状都是B,512,32
# 			sub_Q = Q[:, [i], ...].squeeze(1)       #第一次错误是因为我的squeeze的参数设置为0 这样一来的话，当我们的batchsiez为1的时候也会被压缩
# 			sub_K = K[:, [i], ...].squeeze(1)
# 			sub_V = V[:, [i], ...].squeeze(1)
#
# 			# 每个头的自注意力查询 然后进行缩放
# 			# (B,512,32) mm (B , 32 , 512)
# 			score = torch.bmm(sub_Q, torch.transpose(sub_K, 1, 2)) / torch.sqrt(torch.tensor(sub_Q.size(-1)))
# 			# (B , 512 , 512)
# 			attention_weights = self.softmax(score)
#
# 			# (B , 512 , 512) mm (B , 512 , 32)
# 			weighted_value = torch.bmm(attention_weights, sub_V)
# 			weighted_values.append(weighted_value)
# 		'''
#
#
# 		# torch.tensor(weighted_values)
# 		# # 堆叠到一起 的形状 (B , 512, 256)
# 		# atten_value = torch.stack(weighted_values, dim=2).view(batch_size, point_num, -1)
# 		# 这里还需要把最后得到的注意力特征转置 ， 于原始的特征在第二维对应上
#
# 		return weighted_atten


@BACKBONES.register_module()
class PointNet2SAMSG(BasePointNet):
	"""PointNet2 with Multi-scale grouping.

	Args:
		in_channels (int): Input channels of point cloud.
		num_points (tuple[int]): The number of points which each SA
			module samples.
		radii (tuple[float]): Sampling radii of each SA module.
		num_samples (tuple[int]): The number of samples for ball
			query in each SA module.
		sa_channels (tuple[tuple[int]]): Out channels of each mlp in SA module.
		aggregation_channels (tuple[int]): Out channels of aggregation
			multi-scale grouping features.
		fps_mods (tuple[int]): Mod of FPS for each SA module.
		fps_sample_range_lists (tuple[tuple[int]]): The number of sampling
			points which each SA module samples.
		dilated_group (tuple[bool]): Whether to use dilated ball query for
		out_indices (Sequence[int]): Output from which stages.
		norm_cfg (dict): Config of normalization layer.
		sa_cfg (dict): Config of set abstraction module, which may contain
			the following keys and values:

			- pool_mod (str): Pool method ('max' or 'avg') for SA modules.
			- use_xyz (bool): Whether to use xyz as a part of features.
			- normalize_xyz (bool): Whether to normalize xyz with radii in
			  each SA module.
	"""

	def __init__(self,
	             in_channels = 4,
	             num_points=(2048, 1024, 512, 256),
	             radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
	             num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
	             sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
	                          ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
	                          ((128, 128, 256), (128, 192, 256), (128, 256,
	                                                              256))),
	             aggregation_channels=(64, 128, 256),
	             fps_mods=(('D-FPS'), ('FS'), ('F-FPS', 'D-FPS')),
	             fps_sample_range_lists=((-1), (-1), (512, -1)),
	             dilated_group=(True, True, True),
	             out_indices=(2,),
	             norm_cfg=dict(type='BN2d'),
	             sa_cfg=dict(
		             type='PointSAModuleMSG',
		             pool_mod='max',
		             use_xyz=True,
		             normalize_xyz=False),
	             init_cfg=None):
		super().__init__(init_cfg=init_cfg)
		self.num_sa = len(sa_channels)
		self.out_indices = out_indices
		assert max(out_indices) < self.num_sa
		assert len(num_points) == len(radii) == len(num_samples) == len(
			sa_channels)
		if aggregation_channels is not None:
			assert len(sa_channels) == len(aggregation_channels)
		else:
			aggregation_channels = [None] * len(sa_channels)

		self.SA_modules = nn.ModuleList()
		self.aggregation_mlps = nn.ModuleList()
		sa_in_channel = in_channels - 3  # number of channels without xyz 不包含坐标的特征
		skip_channel_list = [sa_in_channel]

		for sa_index in range(self.num_sa):
			cur_sa_mlps = list(sa_channels[sa_index])
			sa_out_channel = 0
			for radius_index in range(len(radii[sa_index])):
				cur_sa_mlps[radius_index] = [sa_in_channel] + list(
					cur_sa_mlps[radius_index])
				sa_out_channel += cur_sa_mlps[radius_index][-1]

			if isinstance(fps_mods[sa_index], tuple):
				cur_fps_mod = list(fps_mods[sa_index])
			else:
				cur_fps_mod = list([fps_mods[sa_index]])

			if isinstance(fps_sample_range_lists[sa_index], tuple):
				cur_fps_sample_range_list = list(
					fps_sample_range_lists[sa_index])
			else:
				cur_fps_sample_range_list = list(
					[fps_sample_range_lists[sa_index]])

			self.SA_modules.append(
				build_sa_module(
					num_point=num_points[sa_index],
					radii=radii[sa_index],
					sample_nums=num_samples[sa_index],
					mlp_channels=cur_sa_mlps,
					fps_mod=cur_fps_mod,
					fps_sample_range_list=cur_fps_sample_range_list,
					dilated_group=dilated_group[sa_index],
					norm_cfg=norm_cfg,
					cfg=sa_cfg,
					bias=True))
			skip_channel_list.append(sa_out_channel)

			cur_aggregation_channel = aggregation_channels[sa_index]
			if cur_aggregation_channel is None:
				self.aggregation_mlps.append(None)
				sa_in_channel = sa_out_channel
			else:
				self.aggregation_mlps.append(
					ConvModule(
						sa_out_channel,
						cur_aggregation_channel,
						conv_cfg=dict(type='Conv1d'),
						norm_cfg=dict(type='BN1d'),
						kernel_size=1,
						bias=True))
				sa_in_channel = cur_aggregation_channel

	@auto_fp16(apply_to=('points',))
	def forward(self, points):
		"""Forward pass.

		Args:
			points (torch.Tensor): point coordinates with features,
				with shape (B, N, 3 + input_feature_dim).

		Returns:
			dict[str, torch.Tensor]: Outputs of the last SA module.

				- sa_xyz (torch.Tensor): The coordinates of sa features.
				- sa_features (torch.Tensor): The features from the
					last Set Aggregation Layers.
				- sa_indices (torch.Tensor): Indices of the
					input points.
		"""
		xyz, features = self._split_point_feats(points)

		# selfattention_xyz = SelfAttention_xyz(xyz.size(-1))
		# selfattention_feature = SelfAttention_feature(features.size(-2))
		#
		# # 尝试先在cpu上面跑自注意力，然后再把输出张量放到gpu上
		# device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		# device_cpu = torch.device('cpu')
		#
		#
		# # selfattention_xyz.to(device)
		# # selfattention_feature.to(device)
		# # 不能在这进行自注意力 因为太消显存
		# xyz = selfattention_xyz(xyz.to(device_cpu))
		# features = selfattention_feature(features.to(device_cpu))
		# # torch.cuda.empty_cache()
		# xyz = xyz.to(device_gpu)
		# features = features.to(device_gpu)

		batch, num_points = xyz.shape[:2]
		# 索引设置
		indices = xyz.new_tensor(range(num_points)).unsqueeze(0).repeat(
			batch, 1).long()
		# 原始包含全部点的坐标为sa_xyz里面的第一个元素
		sa_xyz = [xyz]
		# 原始包含全部点的特征为sa_features里面的第一个元素
		sa_features = [features]
		# 原始包含全部点的坐标为sa_indices里面的第一个元素
		sa_indices = [indices]

		out_sa_xyz = [xyz]
		out_sa_features = [features]
		out_sa_indices = [indices]

		for i in range(self.num_sa):
			# 进行下采样
			cur_xyz, cur_features, cur_indices = self.SA_modules[i](
				sa_xyz[i], sa_features[i])
			if self.aggregation_mlps[i] is not None:
				cur_features = self.aggregation_mlps[i](cur_features)
			sa_xyz.append(cur_xyz)
			sa_features.append(cur_features)
			sa_indices.append(
				torch.gather(sa_indices[-1], 1, cur_indices.long()))
			if i in self.out_indices:
				out_sa_xyz.append(sa_xyz[-1])
				out_sa_features.append(sa_features[-1])
				out_sa_indices.append(sa_indices[-1])

		'''
		我们最终只对最后一层的重要关键的进行自注意力关注
		'''
		# selfattention_xyz = SelfAttention_xyz(out_sa_xyz[-1].size(-1)).to(
		# 	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		# selfattention_feature = SelfAttention_feature(out_sa_features[-1].size(-2)).to(
		# 	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		# features_transformer = My_Transformer(out_sa_features[-1].size(-2), 4, 1).to(
		# 	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		# selfattention_feature = MultiHead_SelfAttention(out_sa_features[-1].size(-2), 8).to(
		# 	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		# feature应该是256
		# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		# last_xyz = out_sa_xyz.pop(-1)
		# （B , 256 , 512）
		# last_feature = out_sa_features.pop(-1).transpose(1,2)

		# xyz_transformer = My_Transformer_xyz(last_xyz.size(-1),1,64,4).to(device)
		# feature_transformer = My_Transformer_features(last_feature.size(-1),8,512,2).to(device)

		# xyz_atten = xyz_transformer(last_xyz)
		# feature_atten = feature_transformer(last_feature).transpose(1,2)
		# attention_xyz = selfattention_xyz(last_xyz)
		# attention_features = features_transformer(last_feature).transpose(1,2)
		# out_sa_xyz.append(xyz_atten)
		# out_sa_features.append(feature_atten)

		# 输入的tensor的形状为  (batch_size , num_points , num_features)  (B,512,256)
		# attention_feature = selfattention_feature(last_feature)

		# out_sa_xyz.append(attention_xyz)
		# out_sa_features.append(attention_features)

		# TODO
		# 写完了但是出现了梯度消失的情况
		return dict(
			sa_xyz=out_sa_xyz,
			sa_features=out_sa_features,
			sa_indices=out_sa_indices)
