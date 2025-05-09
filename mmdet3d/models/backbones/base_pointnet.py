# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta

import torch
import  torch.nn as nn
from mmcv.runner import BaseModule



'''
自注意力机制
'''
class SelfAttention(nn.Module):
    def __init__(self,input_dim):
        super(SelfAttention, self).__init__()
        # 查询矩阵
        self.query = nn.Linear(input_dim,input_dim)
        # 键矩阵
        self.key = nn.Linear(input_dim,input_dim)
        # 值矩阵
        self.value = nn.Linear(input_dim,input_dim)
        self.sofa_max = nn.Softmax(dim=-1)


    def forward(self,x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # x : (b,n,features) 2*4096*4
        # QK相乘得到相似度得分
        # score : shape(b,4096,4096)
        score = (torch.bmm(query,torch.transpose(key,(1,2)))) / torch.sqrt(x.size(-1)).float()

        attention_weights = self.sofa_max(score)
        weighted_valure = torch.bmm(attention_weights,value)
        out_put = weighted_valure

        return  out_put





class BasePointNet(BaseModule, metaclass=ABCMeta):
    """Base class for PointNet."""

    def __init__(self, init_cfg=None, pretrained=None):
        super(BasePointNet, self).__init__(init_cfg)
        self.fp16_enabled = False
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    @staticmethod
    def _split_point_feats(points):
        """Split coordinates and features of input points.

        Args:
            points (torch.Tensor): Point coordinates with features,
                with shape (B, N, 3 + input_feature_dim).

        Returns:
            torch.Tensor: Coordinates of input points.
            torch.Tensor: Features of input points.
        """



        xyz = points[..., 0:3].contiguous()
        if points.size(-1) > 3:
            # 判断是否除了xyz还有其他特征，如果有进行提取并转换位置
            features = points[..., 3:].transpose(1, 2).contiguous()
        else:
            features = None

        return xyz, features
