import os
import os.path as osp
from glob import glob
import cv2
import argparse
import json
import numpy as np
from tqdm import tqdm

# setup: https://mmpose.readthedocs.io/en/latest/installation.html

def parse_args():
    parser = argparse.ArgumentParser()
    # 添加 root_path 参数，指定输入数据的根目录
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    # 确保设置了 root_path
    assert args.root_path, "Please set root_path."
    return args

# 获取命令行参数
args = parse_args()
root_path = args.root_path

# ---------------------------------------------------------
# 1. 运行 MMPose 进行全身关键点检测
# ---------------------------------------------------------
# 设置可视化和和结果的临时输出目录
output_root = './vis_results'
# 清理旧的输出目录
os.system('rm -rf ' + output_root)

# 确保 MMPose 所需的预训练模型文件存在
assert osp.isfile('./dw-ll_ucoco_384.pth'), 'Please download dw-ll_ucoco_384.pth'
assert osp.isfile('./rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'), 'Please download rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

# 构建并执行 MMPose 运行命令
# 使用 topdown_demo_with_mmdet.py 进行检测
# 参数说明:
# 1. 目标检测配置: demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py
# 2. 目标检测权重: rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
# 3. 姿态估计配置: configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py
# 4. 姿态估计权重: dw-ll_ucoco_384.pth
# --input: 输入图片目录
# --output-root: 输出目录
# --save-predictions: 保存预测结果 (JSON)
cmd = 'python demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py dw-ll_ucoco_384.pth  --input ' + osp.join(root_path, 'frames') + ' --output-root ' + output_root + ' --save-predictions'
print(cmd)
os.system(cmd)

# ---------------------------------------------------------
# 2. 处理并保存关键点数据
# ---------------------------------------------------------
# 创建最终保存关键点的目录
os.makedirs(osp.join(root_path, 'keypoints_whole_body'), exist_ok=True)

# 获取所有帧索引
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(root_path, 'frames', '*.png'))])
# 获取 MMPose 输出的所有 JSON 文件
output_path_list = glob(osp.join(output_root, '*.json'))

for output_path in output_path_list:
    # 从文件名解析帧索引 (文件名格式依赖于 MMPose 输出)
    # 假设格式如 results_FRAME_IDX.json
    frame_idx = int(output_path.split('/')[-1].split('results_')[1][:-5])
    
    with open(output_path) as f:
        out = json.load(f)

    # 选择置信度最高的人体实例
    kpt_save = None
    for i in range(len(out['instance_info'])):
        # 获取关键点坐标 (N, 2)
        xy = np.array(out['instance_info'][i]['keypoints'], dtype=np.float32).reshape(-1,2)
        # 获取关键点置信度 (N, 1)
        score = np.array(out['instance_info'][i]['keypoint_scores'], dtype=np.float32).reshape(-1,1)
        # 合并坐标和置信度 (N, 3) -> [x, y, score]
        kpt = np.concatenate((xy, score),1) # x, y, score
        
        # 这里的策略是选择平均置信度最高的人体
        if (kpt_save is None) or (kpt_save[:,2].mean() < kpt[:,2].mean()):
            kpt_save = kpt
            
    # 保存关键点数据到 JSON 文件
    with open(osp.join(root_path, 'keypoints_whole_body', str(frame_idx) + '.json'), 'w') as f:
        json.dump(kpt_save.tolist(), f)

# ---------------------------------------------------------
# 3. 生成可视化视频
# ---------------------------------------------------------
# 获取可视化结果图片列表
output_path_list = glob(osp.join(output_root, '*.png'))
img_height, img_width = cv2.imread(output_path_list[0]).shape[:2]

# 对生成的图片按帧索引排序
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in glob(osp.join(output_root, '*.png'))])

# 初始化视频写入器
# 视频宽度为原图宽度的两倍 (左: 原图, 右: 关键点可视化结果)
video_save = cv2.VideoWriter(osp.join(root_path, 'keypoints_whole_body.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height)) 

for frame_idx in frame_idx_list:
    # 读取原始图片
    img = cv2.imread(osp.join(root_path, 'frames', str(frame_idx) + '.png'))
    # 读取 MMPose 可视化输出图片
    output = cv2.imread(osp.join(output_root, str(frame_idx) + '.png'))
    
    # 左右拼接
    vis = np.concatenate((img, output),1)
    
    # 添加帧号水印
    vis = cv2.putText(vis, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    
    # 写入通过视频
    video_save.write(vis)

video_save.release()
