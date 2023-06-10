# -*- coding = utf-8 -*-
# @Time : 2023/6/10 17:54
# @Author : Happiness
# @File : test-visualizer.py
# @Software : PyCharm

#### 测试结果可视化

import os
import matplotlib.pyplot as plt
from PIL import Image

fig=plt.figure(figsize=(20,20))  ###画布尺寸


gs = fig.add_gridspec(4,1)  ###四行一列，展示四张

root_path="D:/0.dive into pytorch/openmmlab/mmdetection/projects/cat/work_dirs/cfg-rtmdet-cat/20230610_174654/result/"
image_paths=[filename for filename in os.listdir(root_path)][:4]

for i ,filename in enumerate(image_paths):
    name=os.path.splitext(filename)[0]

    image=Image.open(root_path+filename).convert("RGB")

    ax = fig.add_subplot(gs[i, 0])

    ax.imshow(image)

    plt.xticks([])
    plt.yticks([])

plt.show()
