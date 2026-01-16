import os
import os.path as osp
import cv2
from glob import glob
import argparse

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    # 添加 root_path 参数，用于指定数据的根目录
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()
    # 确保命令行中提供了 root_path 参数，否则报错
    assert args.root_path, "Please set root_path."
    return args

# 获取命令行参数
args = parse_args()
root_path = args.root_path

# ---------------------------------------------------------
# 1. 环境准备与目录初始化
# ---------------------------------------------------------
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
print(f'Initializing COLMAP environments...')
# 定义 COLMAP 运行时的临时工作目录
colmap_path = './colmap_tmp'
# 创建临时目录，如果已存在则忽略
os.makedirs(colmap_path, exist_ok=True)
# 清空临时目录下的所有内容，确保环境干净
os.system('rm -rf ' + osp.join(colmap_path, '*'))

# 创建存放输入图片的子目录
os.makedirs(osp.join(colmap_path, 'images'), exist_ok=True)
# 再次确保图片目录为空
os.system('rm -rf ' + osp.join(colmap_path, 'images', '*'))

# 创建存放稀疏重建结果（Sparse Reconstruction）的子目录
os.makedirs(osp.join(colmap_path, 'sparse'), exist_ok=True)
# 再次确保稀疏重建目录为空
os.system('rm -rf ' + osp.join(colmap_path, 'sparse', '*'))

# ---------------------------------------------------------
# 2. 准备 COLMAP 输入数据
# ---------------------------------------------------------
print(f'Preparing COLMAP input data...')
# 读取帧列表文件（frame_list_all.txt），获取需要处理的帧索引列表
with open(osp.join(root_path, 'frame_list_all.txt')) as f:
    frame_idx_list = [int(x) for x in f.readlines()]

# 遍历所有帧，将图片从原路径复制并转换为 COLMAP 需要的格式
for frame_idx in frame_idx_list:
    # 原始图片路径（假设为 .png 格式）
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    # 读取图片
    img = cv2.imread(img_path)
    # 将图片以 .jpg 格式保存到 COLMAP 的临时图片目录中
    # 命名格式为 image+索引.jpg
    cv2.imwrite(osp.join(colmap_path, 'images', 'image' + str(frame_idx) + '.jpg'), img)

# ---------------------------------------------------------
# 3. 运行 COLMAP 流程
# ---------------------------------------------------------

# 步骤 3.1: 特征提取 (Feature Extraction)
# 提取图像特征并存储到 database.db 中
# --ImageReader.camera_model PINHOLE: 指定相机模型为针孔相机
# --ImageReader.single_camera 1: 假设所有图像使用同一个相机（共享内参）
print(f"Running COLMAP feature extraction...")
cmd = 'colmap feature_extractor --database_path ' + osp.join(colmap_path, 'database.db') + \
      ' --image_path ' + osp.join(colmap_path, 'images') + \
      ' --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1'
res = os.system(cmd)
if res != 0:
    print('Error during COLMAP feature extraction. Terminating.')
    exit(1)

# 步骤 3.2: 特征匹配 (Sequential Matching)
# 对提取的特征进行匹配。使用 sequential_matcher 适用于有序的视频帧序列。
# 如果是无序图片集合，通常使用 exhaustive_matcher。
print(f"Running COLMAP sequential matching...")
cmd = 'colmap sequential_matcher --database_path ' + osp.join(colmap_path, 'database.db')
res = os.system(cmd)
if res != 0:
    print('Error during COLMAP sequential matching. Terminating.')
    exit(1)

# 步骤 3.3: 稀疏重建 (Start Mapper)
# 利用匹配的特征点进行三角化，恢复相机位姿和稀疏点云
# 结果保存到 output_path 指定的目录
print(f"Running COLMAP mapper for sparse reconstruction...")
cmd = 'colmap mapper --database_path ' + osp.join(colmap_path, 'database.db') + \
      ' --image_path ' + osp.join(colmap_path, 'images') + \
      ' --output_path ' + osp.join(colmap_path, 'sparse')
res = os.system(cmd)
if res != 0:
    print('Error during COLMAP mapping. Terminating.')
    exit(1)

# 步骤 3.4: 模型转换 (Model Converter)
# 将 COLMAP 默认生成的二进制模型文件转换为文本格式 (TXT)，便于后续脚本读取
# COLMAP mapper 通常在 sparse 目录下创建一个名为 '0' 的子文件夹存放主模型
print(f"Converting COLMAP model to TXT format...")
cmd = 'colmap model_converter --input_path ' + osp.join(colmap_path, 'sparse', '0') + \
      ' --output_path ' + osp.join(colmap_path, 'sparse', '0') + \
      ' --output_type TXT'
res = os.system(cmd)
if res != 0:
    print('Error during COLMAP model conversion. Terminating.')
    exit(1)

# ---------------------------------------------------------
# 4. 结果整理与清理
# ---------------------------------------------------------
# 在原始 root_path 下创建 sparse 目录，用于存放最终结果
print(f"Organizing COLMAP results...")
os.makedirs(osp.join(root_path, 'sparse'), exist_ok=True)

# 将生成的 TXT 模型文件 (cameras.txt, images.txt, points3D.txt) 移动到目标目录
cmd = 'mv ' + osp.join(colmap_path, 'sparse', '0', '*.txt') + ' ' + osp.join(root_path, 'sparse', '.')
res = os.system(cmd)
if res != 0:
    print('Error during moving COLMAP results. Terminating.')
    exit(1)

# 删除临时的 COLMAP 工作目录，清理空间
cmd = 'rm -rf ' + colmap_path
res = os.system(cmd)
if res != 0:
    print('Error during cleaning up COLMAP temporary files. Terminating.')
    exit(1)