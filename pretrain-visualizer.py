# -*- coding = utf-8 -*-
# @Time : 2023/6/10 15:31
# @Author : Happiness
# @File : pretrain-visualizer.py
# @Software : PyCharm

from mmdet.registry import DATASETS,VISUALIZERS
from mmengine.config import Config
from mmengine.registry import init_default_scope
import matplotlib.pyplot as plt
import os.path as osp

cfg=Config.fromfile("D:/0.dive into pytorch/openmmlab/mmdetection/projects/cat/cfg-rtmdet-cat.py")

init_default_scope(cfg.get("default_scope","mmdet"))

dataset=DATASETS.build(cfg.train_dataloader.dataset)
visualizer=VISUALIZERS.build(cfg.visualizer)
visualizer.dataset_meta=dataset.metainfo

fig = plt.figure()

gs = fig.add_gridspec(1,2)

#只可视化前2张图片
for i in range(2):
    item=dataset[i]

    img=item["inputs"].permute(1,2,0).numpy()
    data_sample=item["data_samples"].numpy()
    gt_instances=data_sample.gt_instances
    img_path=osp.basename(item["data_samples"].img_path)

    gt_bboxes=gt_instances.get("bboxes",None)
    gt_instances.bboxes=gt_bboxes.tensor
    data_sample.gt_instances=gt_instances

    visualizer.add_datasample(
        osp.basename(img_path),
        img,
        data_sample,
        draw_pred=False,
        show=False
    )
    drawed_image=visualizer.get_image()
    ax = fig.add_subplot(gs[0, i])

    ax.imshow(drawed_image)

    plt.xticks([])
    plt.yticks([])

plt.show()

