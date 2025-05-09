# -*- codeing = utf-8 -*-
# @Time : 2023/6/9 20:13
# @Author : 张庭恺
# @File : 生成图片.py
# @Software : PyCharm


from PIL import Image

# 加载源图像和目标图像
target_image = Image.open("../000000.png")
source_image = Image.open("generated_image.png")

# 定义粘贴位置
paste_position = (100, 100)

# 进行粘贴操作
target_image.paste(source_image, paste_position)

# 显示结果图像
target_image.show()
