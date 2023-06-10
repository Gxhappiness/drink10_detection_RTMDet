# drink10_detection_RTMDet




十类饮料目标检测数据集Drink_284
拍摄：张子豪（同济子豪兄）、田文博
标注：张子豪（同济子豪兄）






MS COCO dataset:https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Drink_284_Detection_Dataset/Drink_284_Detection_coco.zip





特征图（feature map）可视化的环境依赖和相关配置参见本人CSDN帖子：https://bbs.csdn.net/topics/615881624









相关模型文件和权重下载链接：https://pan.baidu.com/s/1oc_JLM6oeRKO4pEeVc2Krw?pwd=4gol 
提取码：4gol











配置文件cfg-rtmdet-drink10.py中训练轮次epoch是100，获取到best_coco_bbox_mAP的epoch轮次是38，并且一直到56epoch时验证集性能都没有提升，此时采取了早停政策










![image](https://github.com/Gxhappiness/drink10_detection_RTMDet/assets/95199650/e33433c2-316c-4331-9d84-c997b80444aa)




![image](https://github.com/Gxhappiness/drink10_detection_RTMDet/assets/95199650/eea20cbd-53ce-4707-a61b-ea0de4f84882)
