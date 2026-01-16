import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
from utils.vis import render_mesh
import json
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_one_box(det_output):
    """
    从目标检测输出中提取置信度最高的边界框。
    Args:
        det_output: 目标检测模型的输出，包含 'boxes' 和 'scores'
    Returns:
        max_bbox: 置信度最高的边界框坐标列表 [x_min, y_min, x_max, y_max]，如果没有检测到则返回 None
    """
    max_score = 0
    max_bbox = None

    for i in range(det_output['boxes'].shape[0]):
        bbox = det_output['boxes'][i]
        score = det_output['scores'][i]
        if float(score) > max_score:
            max_bbox = [float(x) for x in bbox]
            max_score = score

    return max_bbox

def parse_args():
    parser = argparse.ArgumentParser()
    # arg: gpu_ids - 指定使用的 GPU ID，例如 "0" 或 "0-1"
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    # arg: root_path - 数据集的根目录路径
    parser.add_argument('--root_path', type=str, dest='root_path')
    args = parser.parse_args()

    # 检查 gpu_ids 是否设置
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    # 处理 GPU ID 范围，例如 "0-3" 转换为 "0,1,2,3"
    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    # 确保设置了 root_path
    assert args.root_path, "Please set root_path."
    return args

# ---------------------------------------------------------
# 1. 初始化设置与模型加载
# ---------------------------------------------------------
args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True
root_path = args.root_path

# 加载 Hand4Whole 模型及其预训练权重 (snapshot)
# 假设模型文件位于当前目录下，名为 snapshot_6.pth.tar
model_path = './snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))

# 初始化模型并设置为评估模式
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()

# ---------------------------------------------------------
# 2. 准备输入数据
# ---------------------------------------------------------
transform = transforms.ToTensor()
# 设置 SMPL-X 参数保存路径
save_path = osp.join(root_path, 'smplx_init')
os.makedirs(save_path, exist_ok=True)

# 获取所有视频帧图片路径
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
img_height, img_width = cv2.imread(img_path_list[0]).shape[:2]

# 初始化视频写入器用于可视化输出
# 输出视频为 smplx_init.mp4，分辨率为两倍宽度（左原图，右渲染结果）
video_save = cv2.VideoWriter(osp.join(root_path, 'smplx_init.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (img_width*2, img_height))

# 按帧索引排序
frame_idx_list = sorted([int(x.split('/')[-1][:-4]) for x in img_path_list])
bbox = None

# 初始化 Faster R-CNN 用于人体检测
det_model = fasterrcnn_resnet50_fpn(pretrained=True).cuda().eval()
det_transform = T.Compose([T.ToTensor()])

# ---------------------------------------------------------
# 3. 逐帧处理
# ---------------------------------------------------------
for frame_idx in tqdm(frame_idx_list):
    # 读取原始图片并转换颜色空间 BGR -> RGB
    img_path = osp.join(root_path, 'frames', str(frame_idx) + '.png')
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    original_img_height, original_img_width = original_img.shape[:2]

    # ---------------------------------------------------------
    # 3.1 人体检测与图像裁剪
    # ---------------------------------------------------------
    det_input = det_transform(original_img).cuda()
    det_output = det_model([det_input])[0]
    bbox = get_one_box(det_output) # xyxy 格式
    
    # 如果未检测到人体，跳过该帧
    if bbox is None:
        continue
    
    # 将 bbox 转换为 xywh 格式
    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]] # xywh
    # 处理 bbox，确保在图像范围内并具有正确的纵横比
    bbox = process_bbox(bbox, original_img_width, original_img_height)
    
    # 生成模型输入的图像 patch
    img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape) 
    img = transform(img.astype(np.float32))/255
    img = img.cuda()[None,:,:,:]

    # ---------------------------------------------------------
    # 3.2 模型推理 (SMPL-X 参数估计)
    # ---------------------------------------------------------
    inputs = {'img': img}
    targets = {}
    meta_info = {}
    with torch.no_grad():
        out = model(inputs, targets, meta_info, 'test')
    
    # 获取预测的 mesh
    mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

    # ---------------------------------------------------------
    # 3.3 可视化与保存视频
    # ---------------------------------------------------------
    # 准备可视化原图 (RGB -> BGR, opencv 格式)
    vis_img = original_img[:,:,::-1].copy()
    
    # 计算用于渲染的相机参数 (focal length 和 principal point)
    # 根据 bbox crop 后的参数变换回原图空间
    focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
    princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
    
    # 渲染 mesh 到原图上
    rendered_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt})
    
    # 拼接原图和渲染结果
    frame = np.concatenate((vis_img, rendered_img),1)
    
    # 在视频帧上添加帧号
    frame = cv2.putText(frame, str(frame_idx), (int(img_width*0.1), int(img_height*0.1)), cv2.FONT_HERSHEY_PLAIN, 2.0, (0,0,255), 3)
    video_save.write(frame.astype(np.uint8))


    # ---------------------------------------------------------
    # 3.4 保存 SMPL-X 参数
    # ---------------------------------------------------------
    # 提取 SMPL-X 各个部分的姿态和形状参数
    # root_pose: 全局旋转 (3,)
    # body_pose: 身体姿态 (63,) -> (21, 3)
    # lhand_pose/rhand_pose: 左右手姿态 (45,) -> (15, 3)
    # jaw_pose: 下颌姿态 (3,)
    # shape: 形状参数 (10,)
    # expr: 表情参数 (10,)
    root_pose = out['smplx_root_pose'].detach().cpu().numpy()[0]
    body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0] 
    lhand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0] 
    rhand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0] 
    jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0] 
    shape = out['smplx_shape'].detach().cpu().numpy()[0]
    expr = out['smplx_expr'].detach().cpu().numpy()[0] 
    
    # 保存参数到 json 文件
    with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
        json.dump({'root_pose': root_pose.reshape(-1).tolist(), \
                'body_pose': body_pose.reshape(-1,3).tolist(), \
                'lhand_pose': lhand_pose.reshape(-1,3).tolist(), \
                'rhand_pose': rhand_pose.reshape(-1,3).tolist(), \
                'leye_pose': [0,0,0],\
                'reye_pose': [0,0,0],\
                'jaw_pose': jaw_pose.reshape(-1).tolist(), \
                'shape': shape.reshape(-1).tolist(), \
                'expr': expr.reshape(-1).tolist()}, f)

