'''
生成虚拟相机参数的脚本。
该脚本会为指定数据集中的每一帧图像生成默认的相机参数，并将其保存为 JSON 文件。
'''
import os
import os.path as osp
import json
import cv2
from glob import glob
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # 添加 root_path 参数，指定数据根目录
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    # 确保提供了 root_path 参数
    assert args.root_path, "Please set root_path."
    return args

# 获取命令行参数
args = parse_args()
root_path = args.root_path

# ---------------------------------------------------------
# 1.获取图像信息
# ---------------------------------------------------------
# 获取 frames 目录下所有的 .png 图片路径
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))

# 读取第一张图片以获取图像尺寸 (高度和宽度)
# 这些尺寸主要用于计算主点 (princpt) 的位置
img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]

# ---------------------------------------------------------
# 2. 生成并保存虚拟相机参数
# ---------------------------------------------------------
# 定义保存相机参数的目录路径
save_root_path = osp.join(root_path, 'cam_params')
# 如果目录不存在，则创建该目录
os.makedirs(save_root_path, exist_ok=True)

# 从文件路径中解析出帧索引 (frame_idx)
# 假设文件名格式为 "索引.png"
frame_idx_list = [int(x.split('/')[-1][:-4]) for x in img_path_list]

# 遍历每一帧，生成对应的默认虚拟相机参数
for frame_idx in frame_idx_list:
    # 定义每个帧对应的 json 文件路径
    with open(osp.join(save_root_path, str(frame_idx) + '.json'), 'w') as f:
        # 构建相机参数字典
        # R: 旋转矩阵，初始化为单位矩阵 (3x3)，表示无旋转
        # t: 平移向量，初始化为零向量 (3,)，表示无平移
        # focal: 焦距，这里硬编码为 (2000, 2000)，表示 fx=2000, fy=2000
        # princpt: 主点坐标 (cx, cy)，设置为图像中心 (width/2, height/2)
        camera_params = {
            'R': np.eye(3).astype(np.float32).tolist(),
            't': np.zeros((3), dtype=np.float32).tolist(),
            'focal': (2000, 2000),
            'princpt': (img_width / 2, img_height / 2)
        }
        
        # 将参数写入 json 文件
        json.dump(camera_params, f)

