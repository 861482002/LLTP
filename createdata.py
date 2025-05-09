# -*- codeing = utf-8 -*-
# @Time : 2023/6/8 20:24
# @Author : 张庭恺
# @File : createdata.py
# @Software : PyCharm

import os
import random
from PIL import Image
import torchvision
from torchvision import transforms
import cv2
from typing import List
img_aug = torchvision.transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.Resize((224,111)),

                                        ])
#
# with open('000000.txt','r') as f:
#     line = f.readline()
#     print(type(line.split()))

# #
# img = Image.open('000000.png')
# # img = cv2.imread('000000.png')
# img = img_aug(img)
# img.show()
# print(img.size)

def zprint(array):
    for i in array:
        i = 5

c = 4

if __name__ == '__main__':
    a = [2,4,1,3]
    a.sort()
    sorted()
    print(a)





