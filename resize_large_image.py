# -*- coding = utf-8 -*-
# @Time : 2023/6/10 20:05
# @Author : Happiness
# @File : futuremap-visualizer.py
# @Software : PyCharm



#####图片分辨率过大会导致特征图可视化很难，甚至会导致程序奔溃，因此我们先将其缩放处理



# import cv2
#
# img=cv2.imread("D:/0.dive into pytorch/openmmlab/mmdetection/projects/drink10/outputs/vis/2.jpg")
# h,w=img.shape[:2]
# resized_img=cv2.resize(img,(640,640))
# cv2.imwrite("resized_2.jpg",resized_img)

import cv2

img=cv2.imread("D:/0.dive into pytorch/openmmlab/mmdetection/projects/drink10/outputs/vis/10.jpg")
h,w=img.shape[:2]
resized_img=cv2.resize(img,(640,640))
cv2.imwrite("resized_10.jpg",resized_img)