# -*- codeing = utf-8 -*-
# @Time : 2023/6/10 19:46
# @Author : 张庭恺
# @File : my_pointnet.py
# @Software : PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TNet(nn.Module):
    def __init__(self, k=3):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2, keepdim=True)

        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        identity = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()

        x = x.view(-1, self.k, self.k) + identity
        return x


class PointNet(nn.Module):
    def __init__(self, k=3):
        super(PointNet, self).__init__()
        self.tnet1 = TNet(k)
        self.tnet2 = TNet(64)

        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        # T-Net 1
        trans = self.tnet1(x)
        x = torch.bmm(trans, x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # T-Net 2
        trans = self.tnet2(x)
        x = torch.bmm(trans, x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)


# 创建一个PointNet实例并打印其结构
pointnet = PointNet()
print(pointnet)

# points = np.fromfile('../000000_8c.bin',dtype=np.float32).reshape(-1,8)
# points_tensor = torch.from_numpy(points)
# points_tensor_xyz = points_tensor[:,:3]
point_tensor = torch.rand((5,3,100))
output = pointnet(point_tensor)
print(output.shape)

