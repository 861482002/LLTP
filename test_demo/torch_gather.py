# -*- codeing = utf-8 -*-
# @Time : 2023/6/14 15:24
# @Author : 张庭恺
# @File : torch_gather.py
# @Software : PyCharm


import torch

input = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

index = torch.tensor([[0, 2],
                      [1, 0]])

output = torch.gather(input, 1, index)
print(output)

