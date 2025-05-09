# -*- codeing = utf-8 -*-
# @Time : 2023/6/19 11:01
# @Author : 张庭恺
# @File : BERT.py
# @Software : PyCharm

import torch
from torch.utils.data import Dataset,DataLoader
import os
from PIL.Image import Image

CLASS = ['kitti','bdd','city']

class customdata(Dataset):
	def __init__(self,data_dir,transform = None):
		super(customdata, self).__init__()
		self.path = data_dir
		self.transform = transform
		self.img_list = os.listdir(self.path)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		img_name = self.img_list[idx]




		return




