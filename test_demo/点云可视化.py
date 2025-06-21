# -*- codeing = utf-8 -*-
# @Time : 2023/6/9 21:37
# @Author : 张庭恺
# @File : 点云可视化.py
# @Software : PyCharm
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# points = np.fromfile('000000.bin',dtype=np.float32).reshape(-1,4)
# x,y,z = points[:,0],points[:,1],points[:,2]
# ax.scatter(x, y, z)
#
# plt.show()

# import torch
#
# x = torch.randn((3,3,4),dtype=torch.float32)
#
# print(x)
# mean = torch.mean(x,dim=-1,keepdim=True)
# print(mean)
'''
装饰器类
'''
def decoreate(fun):
	def wapper():
		print('1')
		a = fun()
		a
		print('3')
	return wapper

@decoreate
def hee():
	print('2')

hee()