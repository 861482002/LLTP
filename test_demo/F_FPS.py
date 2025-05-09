# -*- codeing = utf-8 -*-
# @Time : 2023/6/9 20:15
# @Author : 张庭恺
# @File : F_FPS.py
# @Software : PyCharm

import torch
import torch.nn as nn
import numpy as np
# from mmdet3d.ops import Points_Sampler
from mmdet3d.datasets import PointSample

def farthest_point_sampling_on_features(features, num_points):
    """
    特征最远点采样函数

    参数：
    - features：输入特征，形状为 (batch_size, num_points, num_dims)
    - num_points：采样后的点数

    返回：
    - sampled_features：采样得到的特征，形状为 (batch_size, num_points, num_dims)
    """

    batch_size, total_points, num_dims = features.size()

    # 存储采样后的点索引
    sampled_indices = torch.zeros(batch_size, num_points, dtype=torch.long).to(features.device)
    # 存储已选点到最近选点的距离平方
    distances = torch.ones(batch_size, total_points).to(features.device) * 1e10

    # 随机选择第一个点
    # random_indices = torch.randint(0, total_points, (batch_size,), dtype=torch.long).to(features.device)
    # sampled_indices[:, 0] = random_indices
    # 距离中心点最近的点
    centrol = torch.sum(features,dim=1)/features.size()[1]
    dis = torch.sum((features - centrol)**2,dim=2).squeeze()
    output,index = torch.max(dis,dim=-1)
    sampled_indices[:,0] = index
    for i in range(1,num_points):
        now_point = features[:,sampled_indices[:,i-1],:]

        dis_timely = torch.sum((features - now_point)**2 ,dim=2)

        mask = dis_timely < distances
        distances[mask] = dis_timely[mask]

        output_timely,index_timely = torch.max(distances, dim=-1)
        sampled_indices[:,i] = index_timely


    # for i in range(1, num_points):
    #     last_point = features[:, sampled_indices[:, i - 1], :]
    #     # 计算当前点到已选点的距离平方
    #     dists = torch.sum((features - last_point) ** 2, dim=-1)
    #     # 更新最小距离和对应的点索引
    #     mask = dists < distances
    #     distances[mask] = dists[mask]
    #     sampled_indices[:, i] = torch.max(distances, dim=-1)[1]

    # 根据索引获取采样特征
    sampled_features = torch.gather(features, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, num_dims))

    return sampled_features

if __name__ == '__main__':
    # points_4c = np.fromfile('../data/kitti/training/velodyne_reduced/000000.bin',dtype=np.float32).reshape(-1,4)
    points = np.fromfile('000000_8c.bin',dtype=np.float32).reshape(-1,8)
    points = torch.from_numpy(points)
    points_feature = points[:,3:].view(1,points.shape[0],5)

    sampler = PointSample(4096)
    point_sampled = sampler(points_feature)

    # sampling_on_features = farthest_point_sampling_on_features(points_feature, 4096)

    # print(sampling_on_features.size())
    print(point_sampled.size())


