import os
import os.path as osp
import json
import cv2
import numpy as np
import argparse
from glob import glob
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    # 添加 root_path 参数，指定输入数据的根目录
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    # 确保 root_path 被设置
    assert args.root_path, "Please set root_path."
    return args

# 获取命令行参数
args = parse_args()
root_path = args.root_path

# ---------------------------------------------------------
# 1. 运行 DECA 进行人脸重建
# ---------------------------------------------------------
# 检查 DECA 所需的模型文件是否存在
assert osp.isfile('./data/deca_model.tar'), 'please download deca_model.tar with fetch_data.sh'
assert osp.isfile('./data/FLAME2020/female_model.pkl'), 'please download female_model.pkl with fetch_data.sh'
assert osp.isfile('./data/FLAME2020/male_model.pkl'), 'please download male_model.pkl with fetch_data.sh'
assert osp.isfile('./data/generic_model.pkl'), 'please download generic_model.pkl with fetch_data.sh'

# 设置 DECA 输出结果的临时保存路径
output_save_path = './flame_parmas_out'
# 如果路径已存在，清空并重新创建
os.system('rm -rf ' + output_save_path)
os.makedirs(output_save_path, exist_ok=True)

# 构建并执行 DECA 运行命令
# -i: 输入图像目录
# --saveDepth True: 保存深度图
# --saveObj True: 保存 OBJ 模型
# --rasterizer_type=pytorch3d: 使用 PyTorch3D 光栅化
# --savefolder: 输出保存目录
cmd = 'python demos/demo_reconstruct.py -i ' + osp.join(root_path, 'frames') + ' --saveDepth True --saveObj True --rasterizer_type=pytorch3d --savefolder ' + output_save_path
os.system(cmd)

# ---------------------------------------------------------
# 2. 处理 DECA 输出结果并提取 FLAME 参数
# ---------------------------------------------------------
# 设置提取后的 FLAME 参数保存路径
save_path = osp.join(root_path, 'flame_init', 'flame_params')
os.makedirs(save_path, exist_ok=True)

# 用于存储所有帧的 shape 参数，以便计算平均 shape
flame_shape_param = []
# 获取 DECA 输出的所有 JSON 文件路径
output_path_list = [x for x in glob(osp.join(output_save_path, '*.json'))]

for output_path in output_path_list:
    # 从文件名中解析帧索引 (假设文件名格式为 "索引.json")
    frame_idx = int(output_path.split('/')[-1][:-5])
    
    # 读取 JSON 文件内容
    with open(output_path) as f:
        flame_param = json.load(f)
    
    # 检查 DECA 是否成功检测到人脸 (is_valid)
    if flame_param['is_valid']:
        # 提取 pose 参数：前3维是 global rotation (root_pose)，后3维是 jaw pose
        root_pose, jaw_pose = torch.FloatTensor(flame_param['pose'])[:,:3].view(3), torch.FloatTensor(flame_param['pose'])[:,3:].view(3)
        # 提取 shape 和 expression 参数
        shape = torch.FloatTensor(flame_param['shape']).view(-1)
        expr = torch.FloatTensor(flame_param['exp']).view(-1)
        
        # 收集 shape 参数以计算平均值
        flame_shape_param.append(shape)

        # 转换为 list 格式以便 JSON 序列化
        root_pose, jaw_pose, shape, expr = root_pose.tolist(), jaw_pose.tolist(), shape.tolist(), expr.tolist()
        # 初始化 neck, left eye, right eye pose 为零向量
        neck_pose, leye_pose, reye_pose = [0,0,0], [0,0,0], [0,0,0]
    else:
         # 如果检测无效，相关参数置为 None
         root_pose, jaw_pose, neck_pose, leye_pose, reye_pose, expr, shape = None, None, None, None, None, None, None
    
    # 构建最终的参数字典
    flame_param = {'root_pose': root_pose, 'neck_pose': neck_pose, 'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose, 'expr': expr, 'is_valid': flame_param['is_valid']}
    
    # 保存该帧的参数到指定文件
    with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
        json.dump(flame_param, f)

# 计算所有有效帧的 shape 参数的平均值
flame_shape_param = torch.stack(flame_shape_param).mean(0).tolist()
# 保存平均 shape 参数
with open(osp.join(root_path, 'flame_init', 'shape_param.json'), 'w') as f:
    json.dump(flame_shape_param, f)

# ---------------------------------------------------------
# 3. 整理可视化结果
# ---------------------------------------------------------
# 设置可视化图片的保存路径
save_path = osp.join(root_path, 'flame_init', 'renders')
os.makedirs(save_path, exist_ok=True)

# 获取 DECA 输出的可视化图片 (假设后缀为 _vis.jpg)
vis_path_list = glob(osp.join(output_save_path, '*_vis.jpg'))
for vis_path in vis_path_list:
    # 解析帧索引
    frame_idx = int(vis_path.split('/')[-1].split('_')[0])
    # 将图片复制到目标目录，并重命名为 "帧索引.jpg"
    cmd = 'cp ' + vis_path + ' ' + osp.join(save_path, str(frame_idx) + '.jpg')
    os.system(cmd)


