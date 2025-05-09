# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import Voxelization
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.ops import (PointFPModule, Points_Sampler, QueryAndGroup,
                         gather_points)
import numpy as np
import torch
from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from .. import builder
from ..builder import DETECTORS
from .single_stage import SingleStage3DDetector
from ..transformer import *
from ..ECA_net import ECA_net
from ..CBAM_net import *

# 通道注意力
from ..CA_attention import CA_Atten
# 空间注意力
from ..CBAM_net import spacial_attention
from ..F_FPS import farthest_point_sampling_on_features


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
	r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

	def __init__(self,
	             voxel_layer,
	             voxel_encoder,
	             middle_encoder,
	             backbone,
	             neck=None,
	             bbox_head=None,
	             train_cfg=None,
	             test_cfg=None,
	             init_cfg=None,
	             pretrained=None,
	             trf_indim=4,
	             head_num=1,
	             n_encoder=4,
	             pe_dim=4,
	             ff_dim=64,
	             len=32):
		super(VoxelNet, self).__init__(
			backbone=backbone,
			neck=neck,
			bbox_head=bbox_head,
			train_cfg=train_cfg,
			test_cfg=test_cfg,
			init_cfg=init_cfg,
			pretrained=pretrained)
		self.voxel_layer = Voxelization(**voxel_layer)
		self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
		self.middle_encoder = builder.build_middle_encoder(middle_encoder)
		self.transformer = My_Transformer_features(trf_indim, head_num, ff_dim, n_encoder)
		self.pe = PositionalEncoding_singlevoxel(pe_dim, len)

	# self.ca_atten = CA_Atten(384)

	# 这就是特征提取的阶段，就可以理解为编码器 然后会送到 Detectionhead里面 去解码并得到损失值，然后去反向传播
	def extract_feat(self, points, img_metas=None):
		"""Extract features from points."""
		voxels, num_points, coors = self.voxelize(points)  # 一些格子，每个格子中的点的数量，格子在原始场景中的坐标
		# voxels, num_points, coors = self.voxelize_transformer(points)  # 一些格子，每个格子中的点的数量，格子在原始场景中的坐标
		voxel_features = self.voxel_encoder(voxels, num_points, coors)
		# 上一步会得到一个 (num_voxel , per_voxel_pointsnum,feature) 比如 （6430 * batch_size , 64)
		# 把一个batch里面的点云都cat到一起
		# [6430 * 2 , 14] --> [6430 * 2 , 64]
		'''
		在pillars上面做自注意力机制
		每个pillars之间的注意力
		'''

		# atten_voxel_features = []
		# batch_size = coors[-1, 0].item() + 1
		# device = torch.device('cuda')
		# for i in range(batch_size):
		# 	batch_mask = coors[:, 0] == i
		# 	res_voxel_feature = voxel_features[batch_mask].unsqueeze(0)
		# 	sampler = Points_Sampler([4096],['D-FPS'])
		# 	index = sampler
		# 	down_feature = torch.nn.Conv1d(res_voxel_feature.size(0), res_voxel_feature.size(0),kernel_size=3,stride=2,padding=1,device = device)
		# 	res_voxel_feature = down_feature(res_voxel_feature)
		# 	res_voxel_feature_transformer = My_Transformer_features(64,4,4).to(device)
		# 	atten_feature = res_voxel_feature_transformer(res_voxel_feature)
		# 	atten_voxel_features.append(atten_feature)
		# # voxel__xyz = voxel_features
		# atten_voxel_features = torch.cat(atten_voxel_features,dim=0)
		batch_size = coors[-1, 0].item() + 1

		# pillars_scatter 将得到的体素特点重新分配到伪图像上
		# x = self.middle_encoder(voxel_features, coors, batch_size)
		x = self.middle_encoder(voxel_features, coors, batch_size)
		# second 为主干网络
		# 会得到3层不同尺度的特征
		x = self.backbone(x)
		# 这里经过伪图像特征提取，我们再做注意力机制
		# 接下来是一个fpn层

		# 经过secondfpn 处理后的伪图像 的形状为 batch_size 384 248 216
		if self.with_neck:
			x = self.neck(x)

		return x

	def extract_feat_attention(self, points, img_metas=None):
		"""Extract features from points."""
		# coors : batch_size_num, c,h,w
		voxels, num_points, coors = self.voxelize(points)  # 一些格子，每个格子中的点的数量，格子在原始场景中的坐标
		# voxels: [ batch_size * 6542 , 32 , 4 ]
		# num_points: [batch_size * 6542]
		# coors: [batch_size * 6542 , 4]

		# voxels, num_points, coors = self.voxelize_transformer(points)  # 一些格子，每个格子中的点的数量，格子在原始场景中的坐标
		voxel_features = self.voxel_encoder(voxels, num_points, coors)
		# 上一步会得到一个 (num_voxel , per_voxel_pointsnum,feature) 比如 （6430 * batch_size ,64)
		# 在voxel_encoder里面对体素自注意力

		# # coors[-1, 0].item() + 1  这句话得到的是一共多少个batch
		batch_size = coors[-1, 0].item() + 1
		#
		#
		# features = []
		# for i in range(batch_size):
		# 	mask = coors[:,0] == i
		# 	per_features = voxel_features[mask, :].unsqueeze(0)
		# 	# per_features = voxel_features[mask, :]
		# 	features.append(per_features)
		# 	# 上面的到的形状是  1 , num_voxel , 64
		# attened_features = []
		# # batch_size , voxel_per_baych , features
		# for i , feature in enumerate(features):
		# 	feature_transfomre = My_Transformer_features(feature.size(-1),4,feature.size(0),2).to(device=device_gpu)
		# 	attened_feature = feature_transfomre(feature).squeeze().to(device_gpu)
		# 	attened_features.append(attened_feature)
		# 	# voxel_features = torch.cat([attened_features[0].squeeze(),attened_features[1].squeeze()],dim=0)
		# voxel_features = torch.cat(attened_features,dim=0)

		# pillars_scatter 将得到的体素特点重新分配到伪图像上
		x = self.middle_encoder(voxel_features, coors, batch_size)
		'''
		middle_encoder返回的张量形状为 [N , 64 , 496 , 432 ]
		'''
		# second 为主干网络
		# 会得到3层不同尺度的特征
		x = self.backbone(x)
		# 这里经过伪图像特征提取，我们再做注意力机制
		# 接下来是一个fpn层
		'''
		x:list[3]
		x[0]: 4,64  ,248    ,216
		x[1]: 4,128 ,124    ,108
		x[2]: 4,256 ,62     ,54

		device = torch.device('cuda')
		x_list = []
		for i, sub_x in enumerate(x):
			b, c, h, w = sub_x.size()
			# 先空间后通道
			spa_atten = spacial_attention(kernel_size=7).to(device)
			channels_atte = CA_Atten(c).to(device)
			sub_x_spa = spa_atten(sub_x)
			sub_x_spa_channels = channels_atte(sub_x_spa)

			x_list.append(sub_x_spa_channels)
		'''

		# 经过secondfpn 处理后的伪图像 的形状为 batch_size 384 248 216
		if self.with_neck:
			x = self.neck(x)
		# device = torch.device('cuda')
		# in_feature = x[0].size(1)
		# atten_block = self.ca_atten(x[0]).to(device)

		return x

	# return x

	@torch.no_grad()
	@force_fp32()
	def voxelize_transformer(self, points):
		device = torch.device('cuda')
		"""Apply hard voxelization to points."""
		# 这个num_points应该不是点的数量，而应该是每一帧点云经过体素化后被分割成的体素块的数量
		voxels, coors, num_points = [], [], []
		for res in points:
			# res_voxels:(num_voxel , 32 , 4 )
			# res_coors :(num_voxel , 3)
			# res_num_points :(每个体素块里面的点的数量)

			res_voxels, res_coors, res_num_points = self.voxel_layer(res)
			voxels.append(res_voxels)
			coors.append(res_coors)
			num_points.append(res_num_points)

		# 尝试能不能在voxel后，体素化的过程增加自注意力记住（transformer）
		# 反正无论如何都是在特征提取 而transformer是当下被公认的最好的特征提取器
		# batch_size = len(voxels)
		# voxel的形状 如 ： 6587，32，4

		atten_voxels = []
		for i, voxel in enumerate(voxels):
			# voxel_xyz = voxel[..., :3]
			# voxel_feature = voxel[..., 3:]
			# 空间注意力编码
			# xyz_transformer = My_Transformer_xyz(voxel_xyz.size(-1), 1, 64,n_encoder=4).to(device)
			# 特征注意力编码
			# sampling_points = farthest_point_sampling_on_features(voxel_feature,4096)
			# features_transformer = My_Transformer_features(voxel_feature.size(-1), 1, 4).to(device)
			# atten_xyz = xyz_transformer(voxel_xyz)
			voxel = self.pe(voxel).to(device)
			atten_voxel = self.transformer(voxel).to(device)

			# atten_voxel = torch.cat([voxel_xyz, atten_feature], dim=-1)

			atten_voxels.append(atten_voxel)

		atten_voxels = torch.cat(atten_voxels, dim=0)
		# voxels = torch.cat(voxels, dim=0)
		num_points = torch.cat(num_points, dim=0)
		coors_batch = []
		for i, coor in enumerate(coors):
			coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
			coors_batch.append(coor_pad)
		coors_batch = torch.cat(coors_batch, dim=0)
		return atten_voxels, num_points, coors_batch

	def voxelize(self, points):
		# device = torch.device('cuda')
		"""Apply hard voxelization to points."""
		# 这个num_points应该不是点的数量，而应该是每一帧点云经过体素化后被分割成的体素块的数量
		voxels, coors, num_points = [], [], []
		for res in points:
			# [6587,32,4]  [6587,3] [6587]
			res_voxels, res_coors, res_num_points = self.voxel_layer(res)
			voxels.append(res_voxels)
			coors.append(res_coors)
			num_points.append(res_num_points)

		voxels = torch.cat(voxels, dim=0)
		num_points = torch.cat(num_points, dim=0)
		coors_batch = []
		for i, coor in enumerate(coors):
			coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
			coors_batch.append(coor_pad)
		coors_batch = torch.cat(coors_batch, dim=0)
		# coors_batch 第一维度就是betch的索引
		return voxels, num_points, coors_batch

	def forward_train(self,
	                  points,
	                  img_metas,
	                  gt_bboxes_3d,
	                  gt_labels_3d,
	                  gt_bboxes_ignore=None):
		"""Training forward function.

		Args:
			points (list[torch.Tensor]): Point cloud of each sample.
			img_metas (list[dict]): Meta information of each sample
			gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
				boxes for each sample.
			gt_labels_3d (list[torch.Tensor]): Ground truth labels for
				boxes of each sampole
			gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
				boxes to be ignored. Defaults to None.

		Returns:
			dict: Losses of each branch.
		"""
		'''原始的没有注意力的伪图像特征提取'''
		# x = self.extract_feat(points, img_metas)
		'''经过我们增加了注意力模块的伪图像特征提取'''
		x = self.extract_feat_attention(points, img_metas)
		outs = self.bbox_head(x)
		loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
		losses = self.bbox_head.loss(
			*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
		return losses

	def simple_test(self, points, img_metas, imgs=None, rescale=False):
		"""Test function without augmentaiton."""
		x = self.extract_feat(points, img_metas)
		outs = self.bbox_head(x)
		bbox_list = self.bbox_head.get_bboxes(
			*outs, img_metas, rescale=rescale)
		bbox_results = [
			bbox3d2result(bboxes, scores, labels)
			for bboxes, scores, labels in bbox_list
		]
		return bbox_results

	def aug_test(self, points, img_metas, imgs=None, rescale=False):
		"""Test function with augmentaiton."""
		feats = self.extract_feats(points, img_metas)

		# only support aug_test for one sample
		aug_bboxes = []
		for x, img_meta in zip(feats, img_metas):
			outs = self.bbox_head(x)
			bbox_list = self.bbox_head.get_bboxes(
				*outs, img_meta, rescale=rescale)
			bbox_list = [
				dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
				for bboxes, scores, labels in bbox_list
			]
			aug_bboxes.append(bbox_list[0])

		# after merging, bboxes will be rescaled to the original image size
		merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
		                                    self.bbox_head.test_cfg)

		return [merged_bboxes]
