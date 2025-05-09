# -*- codeing = utf-8 -*-
# @Time : 2023/6/11 20:27
# @Author : 张庭恺
# @File : dataloader.py
# @Software : PyCharm


import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # 对样本进行预处理或其他操作
        # 返回处理后的样本
        return sample

# 假设您有一个数据列表
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 创建自定义数据集
dataset = CustomDataset(data)

# 创建数据加载器
batch_size = 3
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 使用数据加载器进行迭代
for batch in dataloader:
    # 在这里处理批次数据
    print(batch.shape)

