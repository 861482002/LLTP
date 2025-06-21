# -*- codeing = utf-8 -*-
# @Time : 2023/2/11 20:06
# @Author : 张庭恺
# @File : setup_test.py
# @Software : PyCharm

import torch.nn as nn
from thop import profile
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

from mmdet3d.apis import init_model, inference_detector, show_result_meshlab

# 原始的模型
config = 'work_dir_pillars_original21/myconfig.py'
checkpoint = 'work_dir_pillars_original21/latest.pth'

# 我们自己的模型
# config = 'work_dir_pillars_mobilenetv3_80_epoch/myconfig.py'
# checkpoint = 'work_dir_pillars_mobilenetv3_80_epoch/latest.pth'

# input = [torch.rand(1, 64, 496, 432, device=torch.device('cuda')),
#          (torch.rand(1, 64, 248, 216, device=torch.device('cuda')),
#           torch.rand(1, 128, 124, 108, device=torch.device('cuda')),
#           torch.rand(1, 256, 62, 54, device=torch.device('cuda'))),
#          (torch.rand(1, 384, 248, 216, device=torch.device('cuda'), dtype=torch.float32),
#           torch.rand(1, 384, 248, 216, device=torch.device('cuda'), dtype=torch.float32),
#           torch.rand(1, 384, 248, 216, device=torch.device('cuda'), dtype=torch.float32)),
#          torch.rand(20394, 8, device=torch.device('cuda')),
#          (torch.ones(3517, 32, 4, device=torch.device('cuda'), dtype=torch.float32),
#           torch.ones(3517, device=torch.device('cuda'), dtype=torch.float32),
#           torch.zeros(3517, 4, device=torch.device('cuda'), dtype=torch.int)),
#          (torch.rand(3517, 64, device=torch.device('cuda')), torch.rand(3517, 4, device=torch.device('cuda')), 1)]

# second_input = [torch.rand(1,256,200,176,device=torch.device('cuda')),
#          (torch.rand(1,128,200,176,device=torch.device('cuda')),torch.rand(1,256,100,88,device=torch.device('cuda'))),
#          (torch.rand(1,512,200,176,device=torch.device('cuda')),torch.rand(1,512,200,176,device=torch.device('cuda')),torch.rand(1,512,200,176,device=torch.device('cuda'))),
#          torch.rand(20394,8,device=torch.device('cuda')),
#          (torch.rand(16000,32,8,device=torch.device('cuda')),torch.rand(16000,device=torch.device('cuda')),torch.rand(16000,4,device=torch.device('cuda'))),
#          (torch.rand(16000,4,device=torch.device('cuda')),torch.rand(16000,4,device=torch.device('cuda')),1)]
#
# input = [torch.rand(1,16384,4,device=torch.device('cuda')),
#          ]
device = torch.device('cuda')
# pointrcnn_input = dict(
#     sa_xyz = [
#         torch.rand(1,16384,3,device = torch.device('cuda')),
#         torch.rand(1,4096,3,device = torch.device('cuda')),
#         torch.rand(1,1024,3,device = torch.device('cuda')),
#         torch.rand(1,256,3,device = torch.device('cuda')),
#         torch.rand(1,64,3,device = torch.device('cuda')),
#               ],
#     sa_features = [
#         torch.rand(1,1,16384,device = torch.device('cuda')),
#         torch.rand(1,96,4096,device = torch.device('cuda')),
#         torch.rand(1,256,1024,device = torch.device('cuda')),
#         torch.rand(1,512,256,device = torch.device('cuda')),
#         torch.rand(1,1024,64,device = torch.device('cuda')),
#     ],
#     sa_indices = [
#         torch.rand(2, 16384,device = torch.device('cuda')),
#         torch.rand(2, 4096,device = torch.device('cuda')),
#         torch.rand(2, 1024,device = torch.device('cuda')),
#         torch.rand(2, 256,device = torch.device('cuda')),
#         torch.rand(2, 64,device = torch.device('cuda')),
#     ]
# )


model = init_model(config,checkpoint, device='cuda:0')
# count_table = parameter_count_table(model)
# f = FlopCountAnalysis(model,(torch.rand(23124,4),))
# print(f.total())
# backbone_flops = FlopCountAnalysis(model.backbone, inputs=(input[0],))
# voxel_encoder_flops = FlopCountAnalysis(model.voxel_encoder, inputs=(input[4][0], input[4][1], input[4][2]))
# neck_flops = FlopCountAnalysis(model.neck, inputs=(input[1],))
# head_flops = FlopCountAnalysis(model.bbox_head, inputs=(input[2],))

# Flpos_total = backbone_flops.total() + voxel_encoder_flops.total() + neck_flops.total() + head_flops.total()
# print(voxel_encoder_flops.total())

# macs, params = profile(model.backbone,inputs=(input[0],))
# macs3, params3 = profile(model.voxel_encoder,inputs=(input[4][0],input[4][1],input[4][2]))


# macs1, params1 = profile(model.neck,inputs=(dict,))
# print(macs3,params3)
# print(model.voxel_encoder)
# print((sum((macs,macs1)))/1e9)
# print((sum((params,params1)))/1e6)
# macs1, params1 = profile(model.neck,inputs=(input[1],))
# macs2, params2 = profile(model.bbox_head,inputs=(input[2],))
# macs3, params3 = profile(model.voxel_encoder,inputs=(input[4]))
#
#
# print((sum((macs1,macs2,macs3)))/1e9)
# print((sum((params1,params2,params3)))/1e6)

# print(type(model))
pts ='data/kitti_tiny/training/velodyne_reduced/000000.bin'
# pts = '000000.bin'
result,data = inference_detector(model,pts)


show_result_meshlab(data,result,'out',show=True)
