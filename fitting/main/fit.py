import argparse
import numpy as np
import cv2
from config import cfg
import torch
import torch.nn as nn
import json
import os
import os.path as osp
from utils.smpl_x import smpl_x
from utils.flame import flame
from glob import glob
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle
from base import Trainer
from utils.vis import render_mesh
from pytorch3d.io import save_ply
from rich.live import Live
from rich.panel import Panel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject_id', type=str, dest='subject_id')

    args = parser.parse_args()
    assert args.subject_id, "Please set subject ID"
    return args

def rotation_6d_to_axis_angle(x):
    """将6D旋转表示转换为轴角表示"""
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))

def main():
    args = parse_args()
    cfg.set_args(args.subject_id)
    
    # 初始化训练器
    trainer = Trainer()
    trainer._make_batch_generator()
    trainer._make_model()
    
    # register initial flame parameters
    # 注册初始每帧 FLAME 参数。将轴角转换为 6D 旋转表示以优化。
    flame_params = {}
    for frame_idx in trainer.flame_params.keys():
        flame_params[frame_idx] = {}
        # FLAME 头部/面部姿态参数
        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
            # 将参数转换为可优化的 Parameter 对象
            flame_params[frame_idx][key] = nn.Parameter(
                matrix_to_rotation_6d(
                    axis_angle_to_matrix(trainer.flame_params[frame_idx][key].cuda())
                )
            )
        # FLAME 面部表情参数
        flame_params[frame_idx]['expr'] = nn.Parameter(
            trainer.flame_params[frame_idx]['expr'].cuda()
        )
        # FLAME 全局位移参数
        flame_params[frame_idx]['trans'] = nn.Parameter(
            trainer.flame_params[frame_idx]['trans'].cuda()
        )
    # 全局 FLAME 形状参数
    flame_shape = nn.Parameter(trainer.flame_shape_param.float().cuda())

    # register initial smplx parameters
    # 注册初始 SMPL-X 参数
    smplx_params = {}
    for frame_idx in trainer.smplx_params.keys():
        smplx_params[frame_idx] = {}
        for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose']:
            smplx_params[frame_idx][key] = nn.Parameter(matrix_to_rotation_6d(axis_angle_to_matrix(trainer.smplx_params[frame_idx][key].cuda())))
        smplx_params[frame_idx]['trans'] = nn.Parameter(trainer.smplx_params[frame_idx]['trans'].cuda())
        # 共享 FLAME 的头部/面部参数
        for key in ['jaw_pose', 'leye_pose', 'reye_pose', 'expr']:
            smplx_params[frame_idx][key] = flame_params[frame_idx][key] # share
    smplx_shape = nn.Parameter(torch.zeros((smpl_x.shape_param_dim)).float().cuda())
    face_offset = nn.Parameter(torch.zeros((flame.vertex_num,3)).float().cuda())
    joint_offset = nn.Parameter(torch.zeros((smpl_x.joint['num'],3)).float().cuda())
    locator_offset = nn.Parameter(torch.zeros((smpl_x.joint['num'],3)).float().cuda())

    # -------- Training Loop --------
    with Live(Panel("Starting optimization...", title="Optimization Status"), refresh_per_second=1) as live:
        for epoch in range(cfg.end_epoch):
            cfg.set_itr_opt_num(epoch)

            for itr_data, data in enumerate(trainer.batch_generator):
                batch_size = data['img_orig'].shape[0]

                for itr_opt in range(cfg.itr_opt_num):
                    cfg.set_stage(epoch, itr_opt)

                    # optimizer
                    # 配置优化器，根据 epoch 和 iteration 决定优化哪些参数
                    if (epoch == 0) and (itr_opt == 0):
                        # smplx and flame root pose and translatioin
                        # 第一阶段：只优化根姿态和全局位移
                        optimizable_params = []
                        for frame_idx in data['frame_idx']:
                            for key in ['root_pose', 'trans']:
                                optimizable_params.append(smplx_params[int(frame_idx)][key])
                            for key in ['root_pose', 'trans']:
                                optimizable_params.append(flame_params[int(frame_idx)][key])
                        trainer.get_optimizer(optimizable_params)
                    elif ((epoch == 0) and (itr_opt == cfg.stage_itr[0])) or ((epoch > 0) and (itr_opt == 0)):
                        # all parameters
                        # 第二阶段：优化所有相关参数
                        if epoch == (cfg.end_epoch - 1):
                            optimizable_params = [] # do not optimize shared parameters to make per-frame parameters consistent with the shared ones
                        else:
                            optimizable_params = [smplx_shape, flame_shape, face_offset, joint_offset, locator_offset] # all shared parameters
                        for frame_idx in data['frame_idx']:
                            for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose', 'trans']: # jaw_pose, leye_pose, reye_pose, and expr is provided by flame_params
                                optimizable_params.append(smplx_params[int(frame_idx)][key])
                            for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr', 'trans']:
                                optimizable_params.append(flame_params[int(frame_idx)][key])
                        trainer.get_optimizer(optimizable_params)

                    # inputs
                    # 准备模型输入数据for key in ['root_pose', 'body_pose', 'lhand_pose', 'rhand_pose', 'trans']: 
                    # jaw_pose, leye_pose, reye_pose, and expr is provided by flame_params
                    smplx_inputs = {'shape': [smplx_shape for _ in range(batch_size)], 'face_offset': [face_offset for _ in range(batch_size)], 'joint_offset': [joint_offset for _ in range(batch_size)], 'locator_offset': [locator_offset for _ in range(batch_size)]}
                    flame_inputs = {'shape': [flame_shape for _ in range(batch_size)]}
                    for frame_idx in data['frame_idx']:
                        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans']:
                            if key not in smplx_inputs:
                                smplx_inputs[key] = [smplx_params[int(frame_idx)][key]]
                            else:
                                smplx_inputs[key].append(smplx_params[int(frame_idx)][key])
                        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expr', 'trans']:
                            if key not in flame_inputs:
                                flame_inputs[key] = [flame_params[int(frame_idx)][key]]
                            else:
                                flame_inputs[key].append(flame_params[int(frame_idx)][key])
                    for key in smplx_inputs.keys():
                        smplx_inputs[key] = torch.stack(smplx_inputs[key])
                    for key in flame_inputs.keys():
                        flame_inputs[key] = torch.stack(flame_inputs[key])

                    # forwrad
                    # 前向传播计算 Loss
                    trainer.set_lr(itr_opt)
                    trainer.optimizer.zero_grad()
                    loss, out = trainer.model(smplx_inputs, flame_inputs, data, return_output=((epoch==cfg.end_epoch-1) and (itr_opt==cfg.itr_opt_num-1)))
                    loss = {k:loss[k].mean() for k in loss}
                    
                    # backward
                    # 反向传播和参数更新
                    sum(loss[k] for k in loss).backward()
                    trainer.optimizer.step()
                    # print(cfg.result_dir)

                    # log
                    screen = [
                        'epoch %d/%d itr_data %d/%d itr_opt %d/%d:' % (epoch, cfg.end_epoch, itr_data, trainer.itr_per_epoch, itr_opt, cfg.itr_opt_num),
                        'lr: %g' % (trainer.get_lr()),
                        ]
                    screen += ['%s: %.4f' % ('loss_' + k, v.detach()) for k,v in loss.items()]
                    live.update(Panel('\n'.join(screen), title="Optimization Status"))
                # 保存结果（仅在最后一个 epoch）
                if epoch != (cfg.end_epoch-1):
                    continue
                save_root_path = osp.join(cfg.result_dir, 'smplx_optimized')
                os.makedirs(save_root_path, exist_ok=True)
                # 提取输出数据以便保存
                smplx_mesh_cam = out['smplx_mesh_cam'].detach().cpu()
                smplx_mesh_cam_wo_jo = out['smplx_mesh_cam_wo_jo'].detach().cpu()
                smplx_mesh_cam_wo_fo = out['smplx_mesh_cam_wo_fo'].detach().cpu()
                smplx_trans = out['smplx_trans'].detach().cpu().numpy()
                flame_mesh_cam = out['flame_mesh_cam'].detach().cpu()
                smplx_mesh_wo_pose_wo_expr = out['smplx_mesh_wo_pose_wo_expr'].detach().cpu()
                smplx_mesh_wo_pose_wo_expr_wo_fo = out['smplx_mesh_wo_pose_wo_expr_wo_fo'].detach().cpu()
                flame_mesh_wo_pose_wo_expr = out['flame_mesh_wo_pose_wo_expr'].detach().cpu()
                for i in range(batch_size):
                    frame_idx = int(data['frame_idx'][i])
                    
                    # mesh
                    # 保存网格数据 (.ply)
                    if (itr_data == 0) and (i == 0):
                        save_ply(osp.join(save_root_path, 'smplx_wo_pose_wo_expr.ply'), torch.FloatTensor(smplx_mesh_wo_pose_wo_expr[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                        save_ply(osp.join(save_root_path, 'smplx_wo_pose_wo_expr_wo_fo.ply'), torch.FloatTensor(smplx_mesh_wo_pose_wo_expr_wo_fo[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                        save_ply(osp.join(save_root_path, 'flame_wo_pose_wo_expr.ply'), torch.FloatTensor(flame_mesh_wo_pose_wo_expr[i]).contiguous(), torch.IntTensor(flame.face).contiguous())
                    save_path = osp.join(save_root_path, 'meshes')
                    os.makedirs(save_path, exist_ok=True)
                    save_ply(osp.join(save_path, str(frame_idx) + '_smplx.ply'), torch.FloatTensor(smplx_mesh_cam[i]).contiguous(), torch.IntTensor(smpl_x.face).contiguous())
                    save_ply(osp.join(save_path, str(frame_idx) + '_flame.ply'), torch.FloatTensor(flame_mesh_cam[i]).contiguous(), torch.IntTensor(flame.face).contiguous())

                    # render
                    # 渲染结果并保存图像
                    save_path = osp.join(save_root_path, 'renders')
                    os.makedirs(save_path, exist_ok=True)
                    render_smplx = render_mesh(smplx_mesh_cam[i].numpy(), smpl_x.face, {'focal': data['cam_param']['focal'][i].numpy(), 'princpt': data['cam_param']['princpt'][i].numpy()}, data['img_orig'][i].numpy()[:,:,::-1], 1.0)
                    cv2.imwrite(osp.join(save_path, str(frame_idx) + '_smplx.jpg'), render_smplx)

                    # smplx parameter
                    save_path = osp.join(save_root_path, 'smplx_params')
                    os.makedirs(save_path, exist_ok=True)
                    with open(osp.join(save_path, str(frame_idx) + '.json'), 'w') as f:
                        json.dump({'root_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['root_pose'].detach().cpu()).numpy().tolist(), \
                                'body_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['body_pose'].detach().cpu()).numpy().tolist(), \
                                'jaw_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['jaw_pose'].detach().cpu()).numpy().tolist(), \
                                'leye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['leye_pose'].detach().cpu()).numpy().tolist(), \
                                'reye_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['reye_pose'].detach().cpu()).numpy().tolist(), \
                                'lhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['lhand_pose'].detach().cpu()).numpy().tolist(), \
                                'rhand_pose': rotation_6d_to_axis_angle(smplx_params[frame_idx]['rhand_pose'].detach().cpu()).numpy().tolist(), \
                                'expr': smplx_params[frame_idx]['expr'].detach().cpu().numpy().tolist(), \
                                'trans': smplx_trans[i].tolist()}, f)
                    if (itr_data == 0) and (i == 0):
                        # shape parameter
                        with open(osp.join(save_root_path, 'shape_param.json'), 'w') as f:
                            json.dump(smplx_shape.detach().cpu().numpy().tolist(), f)
                        # face offset
                        _face_offset = smpl_x.get_face_offset(face_offset[None,:,:])[0]
                        with open(osp.join(save_root_path, 'face_offset.json'), 'w') as f:
                            json.dump(_face_offset.detach().cpu().numpy().tolist(), f)
                        # joint offset
                        _joint_offset = smpl_x.get_joint_offset(joint_offset[None,:,:])[0]
                        with open(osp.join(save_root_path, 'joint_offset.json'), 'w') as f:
                            json.dump(_joint_offset.detach().cpu().numpy().tolist(), f)
                        # locator offset
                        # 生成视频展示优化结果
                        _locator_offset = smpl_x.get_locator_offset(locator_offset[None,:,:])[0]
                        with open(osp.join(save_root_path, 'locator_offset.json'), 'w') as f:
                            json.dump(_locator_offset.detach().cpu().numpy().tolist(), f)

    # video
    save_path = osp.join(save_root_path, '..', 'smplx_optimized.mp4')
    video_shape = cv2.imread(glob(osp.join(save_root_path, 'renders', '*.jpg'))[0]).shape[:2] # height, width
    video_out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_shape[1]*2, video_shape[0]))
    frame_idx_list = sorted([int(x.split('/')[-1].split('_')[0]) for x in glob(osp.join(save_root_path, 'renders', '*_smplx.jpg'))])
    for frame_idx in frame_idx_list:
        orig_img_path = trainer.trainset_loader.img_paths[frame_idx]
        orig_img = cv2.imread(orig_img_path)
        render = cv2.imread(osp.join(save_root_path, 'renders', str(frame_idx) + '_smplx.jpg'))
        out = np.concatenate((orig_img, render),1)
        cv2.putText(out, str(frame_idx), (int(0.02*render.shape[1]), int(0.1*render.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 2.5, [51,51,255], 3, 2) # write frame index
        video_out.write(out)
    video_out.release()



if __name__ == "__main__":
    main()
