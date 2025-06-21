# -*- codeing = utf-8 -*-
# @Time : 2023/6/13 15:03
# @Author : 张庭恺
# @File : 白天到夜间图片.py
# @Software : PyCharm

import cv2

# 加载白天图片
day_image = cv2.imread('000000.png')

# 调整亮度和对比度
adjusted_image = cv2.convertScaleAbs(day_image, alpha=0.3, beta=50)

# 添加色调和色温
# 这里假设你已经实现了相应的函数来调整色彩和色温

# 显示转换后的图片
cv2.imshow('Night Image', adjusted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


