# -*- codeing = utf-8 -*-
# @Time : 2023/6/17 18:56
# @Author : 张庭恺
# @File : test.py
# @Software : PyCharm


import torch
from torch import nn

avg_pool = nn.AdaptiveAvgPool2d((1,1))

avg_pool1 = nn.AdaptiveAvgPool2d(1)
pseudo_img = torch.randn(4,64,256,256)

out = avg_pool(pseudo_img)
out1 = avg_pool1(pseudo_img)

print(out.shape)
print(out1.shape)