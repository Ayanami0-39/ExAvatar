'''
unwrap.py

文件说明：
- 将每帧从数据集中裁切出的面部图像投影到 FLAME 的 UV 贴图空间，
  并对所有帧的投影结果按像素位置累计求平均，输出最终合成的面部纹理图
  以及对应的覆盖掩码。

用法：
    python unwrap.py --subject_id <subject_id>

输入要求（目录结构示例）：
    ../data/<cfg.dataset>/data/<subject_id>/smplx_optimized/
      - meshes/<frame_idx>_flame.ply         # 每帧的 FLAME 网格（顶点）
      - smplx_params/<frame_idx>.json        # 每帧 SMPL-X 参数（expr/trans）
      - shape_param.json, joint_offset.json, locator_offset.json  # subject 级参数

输出：
    <cfg.result_dir>/unwrapped_textures/face_texture.png
    <cfg.result_dir>/unwrapped_textures/face_texture_mask.png

注意事项：
- 脚本基于 repo 中的 `Trainer().batch_generator` 提供的人脸裁切图像与相机参数进行投影，
  需要保证 `Trainer` 与模型的输入格式兼容。脚本会优先使用 GPU（若可用）。
- 为避免数值问题，最终平均时会对除法加上一个小常数 1e-4。
'''

import argparse
import numpy as np
import cv2
from config import cfg
import torch
import json
import os
import os.path as osp
from utils.smpl_x import smpl_x
from utils.flame import flame
from tqdm import tqdm
from base import Trainer
from pytorch3d.io import load_ply
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')
    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x

def main():
    args = parse_args()
    cfg.set_args(args.subject_id, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_path = osp.join('..', 'data', cfg.dataset, 'data', cfg.subject_id, 'smplx_optimized')
    print(f"Processing subject: {cfg.subject_id} at {root_path}")
    print(f"Using device: {device}")

    # 构建 Trainer，并创建数据生成器与模型实例
    # Trainer 内部负责构造 batch_generator，提供 img_face / cam_param_* / frame_idx 等字段
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    model = getattr(trainer.model, "module", trainer.model)
    model.to(device)
    model.eval()

    param_list = []
    for json_name in ['shape_param.json', 'joint_offset.json', 'locator_offset.json']:
        try:
            with open(osp.join(root_path, json_name)) as f:
                param = torch.FloatTensor(json.load(f))
            param_list.append(param)
        except Exception:
            param_list.append(None)
    shape_param, joint_offset, locator_offset = param_list

    # eye keypoint indices
    try:
        le_idx = smpl_x.kpt['name'].index('L_Eye')
        re_idx = smpl_x.kpt['name'].index('R_Eye')
    except Exception:
        le_idx, re_idx = None, None
    print(f"Left eye index: {le_idx}, Right eye index: {re_idx}")

    face_texture_save = None
    face_texture_mask_save = None

    with torch.no_grad():
        for itr, data in enumerate(tqdm(trainer.batch_generator)):
            # move data to device
            for k in list(data.keys()):
                if torch.is_tensor(data[k]):
                    data[k] = data[k].to(device)
                elif isinstance(data[k], dict):
                    for kk in list(data[k].keys()):
                        if torch.is_tensor(data[k][kk]):
                            data[k][kk] = data[k][kk].to(device)
            # 当前批次大小（人脸图像数量）
            batch_size = data['img_face'].shape[0]

            # 加载当前批次对应的 FLAME 网格和 SMPL-X 参数
            flame_mesh_cam_list = []
            smplx_inputs = {'shape': [], 'expr': [], 'trans': [], 'joint_offset': [], 'locator_offset': []}
            valid_frames = [True] * batch_size

            for i in range(batch_size):
                frame_idx = int(data['frame_idx'][i].item() if torch.is_tensor(data['frame_idx'][i]) else data['frame_idx'][i])
                ply_path = osp.join(root_path, 'meshes', str(frame_idx) + '_flame.ply')
                try:
                     # 读取该帧的 FLAME mesh
                    flame_mesh_cam, _ = load_ply(ply_path)
                    flame_mesh_cam_list.append(flame_mesh_cam)
                except Exception:
                    valid_frames[i] = False
                    flame_mesh_cam_list.append(torch.zeros((len(flame.vert), 3)))  # placeholder

                smplx_inputs['shape'].append(shape_param)
                smplx_inputs['joint_offset'].append(joint_offset)
                smplx_inputs['locator_offset'].append(locator_offset)

                # load per-frame smplx params (expr / trans)
                try:
                    with open(osp.join(root_path, 'smplx_params', str(frame_idx) + '.json')) as f:
                        smplx_param = json.load(f)
                    smplx_inputs['expr'].append(torch.FloatTensor(smplx_param.get('expr', [0.0])))
                    smplx_inputs['trans'].append(torch.FloatTensor(smplx_param.get('trans', [0.0])))
                except Exception:
                    smplx_inputs['expr'].append(torch.zeros(1))
                    smplx_inputs['trans'].append(torch.zeros(1))

            # 将每帧 FLAME 顶点列表堆叠为 (B, V, 3)
            flame_mesh_cam = torch.stack(flame_mesh_cam_list).to(device)
            # 将 smplx_inputs 的每个项堆叠为 (B, ...) 
            smplx_inputs = {k: torch.stack(v).to(device) for k, v in smplx_inputs.items()}

            if 'smplx_param' not in data:
                data['smplx_param'] = {}
            # 把组装好的 smplx 参数填回到 data['smplx_param']（模型期望的输入结构）
            data['smplx_param']['shape'] = smplx_inputs['shape'].clone().detach()
            data['smplx_param']['expr'] = smplx_inputs['expr'].clone().detach()
            data['smplx_param']['trans'] = smplx_inputs['trans'].clone().detach()
            data['smplx_param']['joint_offset'] = smplx_inputs['joint_offset'].clone().detach()
            data['smplx_param']['locator_offset'] = smplx_inputs['locator_offset'].clone().detach()

            # 基于当前 smplx 参数和投影相机参数计算 SMPL-X 顶点与关键点（用于可见性判定）
            smplx_mesh_cam_init, smplx_kpt_cam_init, _, _ = model.get_smplx_coord(data['smplx_param'], data['cam_param_proj'], use_face_offset=False)

            # 可见性检测：使用面部顶点与左右眼关键点判断面部是否在视野内
            if le_idx is not None and re_idx is not None:
                face_valid = model.check_face_visibility(
                    smplx_mesh_cam_init[:, smpl_x.face_vertex_idx, :],
                    smplx_kpt_cam_init[:, le_idx, :],
                    smplx_kpt_cam_init[:, re_idx, :])
            else:
                # no eye keypoints, assume all invalid
                face_valid = torch.zeros((batch_size,), dtype=torch.float32, device=device)

            # combine with data flame_valid if present
            if 'flame_valid' in data:
                face_valid = face_valid * data['flame_valid']

            # 将人脸裁切图像投影到 FLAME 的 UV 空间
            # model.xy2uv 的常见输入：
            #  - img_face: (B, C, H, W) 的人脸图像张量
            #  - flame_mesh_cam: (B, V, 3) 的每帧顶点（相机坐标）
            #  - flame.face: FLAME 模型的三角形拓扑
            #  - cam_param_face: 人脸裁切时的相机参数（可能为 dict 或张量）
            # 输出：
            #  - face_texture: (B, C, UV_H, UV_W) 展开到 UV 空间的颜色贡献
            #  - face_texture_mask: (B, 1, UV_H, UV_W) 每帧覆盖掩码
            face_texture, face_texture_mask = model.xy2uv(
                data['img_face'].to(device),    # 人脸裁切图像
                flame_mesh_cam,                 # 对应的 FLAME 顶点（每帧）
                flame.face,                     # FLAME 三角面拓扑（顶点索引）
                data['cam_param_face'])         # 人脸相机参数
            
            # 使用全局 UV mask 屏蔽非面部区域，并按帧的 face_valid 掩码屏蔽不可用帧
            face_texture = face_texture * flame.uv_mask[None, None, :, :] * face_valid[:, None, None, None]
            face_texture_mask = face_texture_mask * flame.uv_mask[None, None, :, :] * face_valid[:, None, None, None]

            # 将当前批次在 UV 空间的贡献求和并转换为 numpy，准备累加到全局贴图
            ft_np = face_texture.sum(0).detach().cpu().numpy()
            fm_np = face_texture_mask.sum(0).detach().cpu().numpy()

            # 第一次累加时初始化累加缓冲区
            if face_texture_save is None:
                face_texture_save = np.zeros_like(ft_np, dtype=np.float64)
            if face_texture_mask_save is None:
                face_texture_mask_save = np.zeros_like(fm_np, dtype=np.float64)

            # 将当前批次的贡献累加到全局缓冲区
            face_texture_save += ft_np
            face_texture_mask_save += fm_np

    if face_texture_save is None or face_texture_mask_save is None:
        raise RuntimeError("No valid face textures were generated.")

    # 计算最终的面部纹理图（除以掩码以获得平均值），并裁剪到有效范围 [0, 1]
    face_texture = face_texture_save / (face_texture_mask_save + 1e-4)
    face_texture = np.clip(face_texture, 0.0, 1.0)
    # 适配 OpenCV 格式 (C,H,W) -> (H,W,C), RGB -> BGR, [0,1] -> [0,255]
    face_texture_img = (face_texture.transpose(1, 2, 0)[:, :, ::-1] * 255.0).astype(np.uint8)

    # 计算最终的面部纹理掩码（二值化）
    face_texture_mask = (face_texture_mask_save > 0).astype(np.uint8)
    # 适配 OpenCV 格式 (C,H,W) -> (H,W,C), RGB -> BGR, [0,1] -> [0,255]
    face_texture_mask_img = (face_texture_mask.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)

    save_root_path = osp.join(cfg.result_dir, 'unwrapped_textures')
    os.makedirs(save_root_path, exist_ok=True)
    cv2.imwrite(osp.join(save_root_path, 'face_texture.png'), face_texture_img)
    cv2.imwrite(osp.join(save_root_path, 'face_texture_mask.png'), face_texture_mask_img)

if __name__ == "__main__":
    main()