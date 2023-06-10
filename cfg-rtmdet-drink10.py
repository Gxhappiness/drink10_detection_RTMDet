# -*- coding = utf-8 -*-
# @Time : 2023/6/10 14:22
# @Author : Happiness
# @Software : PyCharm



#base是coco 80类配置
_base_ = 'D:/0.dive into pytorch/openmmlab/mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'

data_root="D:/0.dive into pytorch/openmmlab/mmdetection/data/Drink_284_Detection_coco/"

# 非常重要
metainfo = {
    # 类别名——注意classes需要的是一个tuple元组，因此即使是单类，也要像下面这样写
    'classes': ('cola',"pepsi","sprite","fanta","spring","ice","scream","milk","red","king",),
    'palette': [
        (220, 20, 60),(220, 220, 60),(220, 20, 160),(220, 120, 60),(220, 200, 60),
        (220, 210, 60),(220, 20, 60),(220, 20, 0),(220, 20, 220),(220, 20, 100),
    ]
}
num_classes = 10

# 训练 100 epoch
max_epochs = 100
#一般训练单卡的batc size=16,可根据自己电脑修改
train_batch_size_per_gpu = 8
# 可以根据自己的电脑修改
train_num_workers = 8

# 验证集 batch size 为 2
val_batch_size_per_gpu = 2
val_num_workers = 4

#RTMDET继承了YOLOX,训练过程分为2个stage,在第二个stage时会更换数据增强策略（pipeline），下面是第二个stage共训练的轮次
num_epochs_stage2=10

#batch改变了，学习率也要跟着改变，二者是正比例相关，原本的base配置中8xb32代表8卡*32的学习率是0.004，我们是单卡8，所以学习率要缩小32倍
base_lr=0.004/32

#采用coco 80类预训练权重
load_from="D:/Data/torch-model/hub/checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"

model = dict(

    backbone=dict(),
    # 不要忘记修改 num_classes
    bbox_head=dict(dict(num_classes=num_classes)))

# 数据集不同，dataset 输入参数也不一样
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    pin_memory=False,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train_coco.json',
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val_coco.json',
        data_prefix=dict(img='images/')))


###不区分验证集和测试集
test_dataloader = val_dataloader

#默认的学习率调度器是warmup 1000（也就是1000次启动一次重启（warmup）策略）
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=100),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,  # max_epoch 也改变了
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

######  optimizer
optim_wrapper = dict(optimizer=dict(lr=base_lr))

# 第二 stage 切换 pipeline 的 epoch 时刻也改变了
_base_.custom_hooks[1].switch_epoch = max_epochs - num_epochs_stage2

val_evaluator = dict(ann_file=data_root + 'val_coco.json')

#不区分验证机和测试集
test_evaluator = val_evaluator

# 一些打印设置修改
default_hooks = dict(
    ### 每隔多少个epoch保存一次模型权重
    checkpoint=dict(interval=1, max_keep_ckpts=2, save_best='auto'),  # 同时保存最好性能权重

    logger=dict(type='LoggerHook', interval=10)) #每隔10次迭代打印一次日志

train_cfg = dict(max_epochs=max_epochs, val_interval=1) #每隔几次epoch做一次验证