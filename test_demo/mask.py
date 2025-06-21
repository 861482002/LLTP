# -*- codeing = utf-8 -*-
# @Time : 2023/6/14 21:44
# @Author : 张庭恺
# @File : mask.py
# @Software : PyCharm


from mmdet3d.ops import (PointFPModule, Points_Sampler, QueryAndGroup,
                         gather_points)
import numpy as np
import torch

device = torch.device('cuda')
points = np.fromfile('000000.bin',dtype=np.float32).reshape(1,-1,4)
print(points.shape)
points = torch.from_numpy(points)
points = torch.tensor(points,device = device)
print(points[:,:,3:].permute(0,2,1).shape)

Sampler = Points_Sampler([4096],['D-FPS'])
print(Sampler)
index = Sampler(points[:,:,:3],points[:,:,3:].permute(0,2,1))
print(index)
new_points = gather_points(points[:,:,0:3].transpose(1,2).contiguous(),index)
print(new_points.shape)

