# -*- coding = utf-8 -*-
# @Time : 2023/6/9 14:25
# @Author : Happiness
# @File : coco.json-visualizer.py
# @Software : PyCharm



#### 环境检测和查看


# from mmengine.utils import get_git_hash
# from mmengine.utils.dl_utils import collect_env as collect_base_env
#
# import mmdet
#
# #环境信息收集和打印
# def collect_env():
#     #### 环境信息收集
#     env_info=collect_base_env()
#     env_info["MMDetection"]=f"{mmdet.__version__}+{get_git_hash()[:7]}"
#     return env_info
#
#
# if __name__ == "__main__":
#     for name , val in collect_env().items():
#         print(f"{name}:{val}")





######  数据集可视化

# import os
# import matplotlib.pyplot as plt
# from PIL import Image
#
# original_images=[]
# images=[]
# texts=[]
# plt.figure(figsize=(16,5))


#先取前八张图片出来可视化
# image_paths=[filename for filename in os.listdir("D:/0.dive into pytorch/openmmlab/mmdetection/data/cat_dataset/images/")][:8]
#
# for i ,filename in enumerate(image_paths):
#     name=os.path.splitext(filename)[0]
#
#     image=Image.open("D:/0.dive into pytorch/openmmlab/mmdetection/data/cat_dataset/images/"+filename).convert("RGB")
#
#     plt.subplot(2,4,i+1)
#     plt.show(image)
#     plt.title(f"{filename}")
#     plt.xticks([])
#     plt.yticks([])
#
# plt.tight_layout()





###### COCO.json可视化

from pycocotools.coco import COCO
import numpy as np
import os.path as osp
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image
import matplotlib.pyplot as plt

def apply_exif_orientation (image):
    _EXIF_ORIENT = 274
    if not hasattr(image, 'getexif'):
        return image
    try:
        exif = image.getexif ()
    except Exception:
        exif = None
    if exif is None:
        return image
    orientation = exif.get (_EXIF_ORIENT)
    method = {
        2: Image. FLIP_LEFT_RIGHT,
        3: Image. ROTATE_180,
        4: Image. FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)
    if method is not None:
        return image.transpose(method)
    return image

def show_bbox_only(coco, anns, show_label_bbox = True, is_filling = True):
    """Show bounding box of annotations Only."""
    if len(anns) == 0:
        return
    ax = plt.gca()
    ax.set_autoscale_on(False)
    image2color = dict()
    for cat in coco.getCatIds():
        image2color[cat] = (np.random.random((1, 3)) * 0.7 + 0.3).tolist()[0]
    polygons = []
    colors = []
    for ann in anns:
        color = image2color[ann['category_id']]
        bbox_x, bbox_y, bbox_w, bbox_h = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y + bbox_h],
            [bbox_x + bbox_w, bbox_y + bbox_h], [bbox_x + bbox_w, bbox_y]]
        polygons.append(Polygon(np.array(poly).reshape((4, 2))))
        colors.append(color)

        if show_label_bbox:
            label_bbox = dict(facecolor=color)
        else:
            label_bbox = None
    ax.text(
        bbox_x, bbox_y,
        "%s" % (coco. loadCats (ann ['category_id'])[0]['name']),
        color='white',
        bbox=label_bbox)
    if is_filling:
        p = PatchCollection(
            polygons, facecolor=colors, linewidths=0, alpha=0.4)
        ax.add_collection(p)
    p = PatchCollection(
            polygons, facecolor='none', edgecolors=colors, linewidths=2)
    ax.add_collection(p)


coco = COCO('D:/0.dive into pytorch/openmmlab/mmdetection/data/Drink_284_Detection_coco/val_coco.json')
image_ids = coco.getImgIds()
np.random.shuffle(image_ids)


fig = plt.figure()

gs = fig.add_gridspec(1,4)

# 只可视化4张图片
for i in range(4):
    image_data = coco.loadImgs(image_ids[i])[0]
    image_path = osp.join('D:/0.dive into pytorch/openmmlab/mmdetection/data/Drink_284_Detection_coco/images/', image_data['file_name'])
    annotation_ids = coco.getAnnIds(
            imgIds=image_data['id'], catIds=[], iscrowd=0)
    annotations = coco.loadAnns(annotation_ids)

    ax = fig.add_subplot(gs[0, i])
    image = Image.open(image_path).convert("RGB")


### 这行代码很关键，不则可能图片和标签对不上
    image = apply_exif_orientation(image)

    ax.imshow(image)
    show_bbox_only(coco, annotations)


    show_bbox_only(coco, annotations)
    plt.xticks([])
    plt.yticks([])

plt.show()


