# -*- codeing = utf-8 -*-
# @Time : 2023/7/5 22:48
# @Author : 张庭恺
# @File : read_coco.py
# @Software : PyCharm


import torchvision
from PIL import ImageDraw
coco = torchvision.datasets.CocoDetection(root='D:\\A_coco_dataset\\val2017\\val2017',
                                          annFile='D:\\A_coco_dataset\\annotations_trainval2017\\annotations\\instances_val2017.json',
                                          )
print(coco)
cats = coco.coco.cats
classes = []
list_idx = cats.keys()
for i in list_idx:
    classes.append(cats[i]['name'])

print(classes)
image , anno= coco[0]
print(anno)
# category_di
img_handler = ImageDraw.ImageDraw(image)
print(len(anno))
for tangle in anno:
    x_min,y_min,width,heigth = tangle['bbox']
    img_handler.rectangle(((x_min,y_min),(x_min+width,y_min+heigth)))
    cat = cats[tangle['category_id']]['name']
    img_handler.text((x_min+10,y_min-10),text=cat,fill='red')

image.show()



