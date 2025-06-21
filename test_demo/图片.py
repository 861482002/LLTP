# -*- codeing = utf-8 -*-
# @Time : 2023/6/9 20:10
# @Author : 张庭恺
# @File : 图片.py
# @Software : PyCharm

from PIL import Image, ImageDraw

# 创建新的图像
image = Image.new("RGB", (300, 300), "white")

# 创建绘图对象
draw = ImageDraw.Draw(image)

# 绘制一个矩形
draw.rectangle([(50, 50), (250, 250)], fill="blue",outline='black')

# 保存图像
image.save("generated_image.png")

# 显示图像
image.show()

