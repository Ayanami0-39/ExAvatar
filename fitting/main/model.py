"""
model.py

本文件实现用于训练与推理的核心 `Model` 类，封装了：
- SMPL-X / FLAME 模型调用的封装（坐标计算与投影）
- 若干损失函数的组合（用于优化人脸/身体/偏移项）
- 将网络层（如 XY->UV 展开）与物理先验/正则化整合在一起

主要接口：
- `Model.forward(smplx_inputs, flame_inputs, data, return_output)`：计算损失并可选返回可视化输出
- `get_smplx_coord`, `get_flame_coord`：将参数转为相机空间坐标并投影为 2D
- `xy2uv` 层用于将人脸图像展开到 FLAME 的 UV 空间（在构造时注入）

使用注意：
- 本模块依赖外部定义的 `smpl_x`、`flame`、以及 `nets` 中的层与损失实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.layer import XY2UV, get_face_index_map_uv
from nets.loss import CoordLoss, PoseLoss, LaplacianReg, EdgeLengthLoss, FaceOffsetSymmetricReg, JointOffsetSymmetricReg
from utils.smpl_x import smpl_x
from utils.flame import flame
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import math
from config import cfg

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # XY->UV 展开层：把人脸图像映射到 FLAME 的 UV 空间
        self.xy2uv = XY2UV(flame.vertex_uv, flame.face_uv, cfg.uvmap_shape)

        # 深复制外部提供的 SMPL-X / FLAME 层（保持独立实例以免修改全局对象）
        self.smplx_layer = copy.deepcopy(smpl_x.layer).cuda()
        self.flame_layer = copy.deepcopy(flame.layer).cuda()

        # 损失函数组件（具体实现放在 nets.loss）
        self.coord_loss = CoordLoss()
        self.pose_loss = PoseLoss()
        self.lap_reg = LaplacianReg(flame.vertex_num, flame.face)
        self.edge_length_loss = EdgeLengthLoss(flame.face)
        self.face_offset_sym_reg = FaceOffsetSymmetricReg()
        self.joint_offset_sym_reg = JointOffsetSymmetricReg()

    def process_input_smplx_param(self, smplx_param):
        '''
        处理 smplx 参数，把6D旋转转换为旋转角
        Param:
            smplx_param
        '''
        out = {}

        # rotation 6d -> axis angle
        # 输入的旋转使用 6D 表示，这里将其转换为 axis-angle 形式以便后续层接受
        for key in ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(smplx_param[key]))

        # others
        # 其余参数直接传递：平移、表情、形状、以及各种偏移项
        out['trans'] = smplx_param['trans']
        out['expr'] = smplx_param['expr']
        out['shape'] = smplx_param['shape']
        out['face_offset'] = smplx_param['face_offset']
        out['joint_offset'] = smplx_param['joint_offset']
        out['locator_offset'] = smplx_param['locator_offset']
        return out

    def process_input_flame_param(self, flame_param):
        '''
        处理 flame 参数，把6D旋转转换为旋转角
        '''
        out = {}

        # rotation 6d -> axis angle
        # 将 FLAME 的 6D 旋转输入也转换为 axis-angle 供 FLAME 层使用
        for key in ['root_pose', 'neck_pose', 'jaw_pose', 'leye_pose', 'reye_pose']:
            out[key] = matrix_to_axis_angle(rotation_6d_to_matrix(flame_param[key]))

        # others
        # 直接传递其他标量/向量参数
        out['trans'] = flame_param['trans']
        out['expr'] = flame_param['expr']
        out['shape'] = flame_param['shape']
        return out

    def get_smplx_coord(self, smplx_param, cam_param, use_pose=True, use_expr=True, use_face_offset=True, use_joint_offset=True, use_locator_offset=True, root_rel=False):
        '''
        获取 SMPL-X 模型在相机坐标系下的网格与关键点坐标，并可选进行 2D 投影
        '''
        batch_size = smplx_param['root_pose'].shape[0]
       
        if use_pose:
            root_pose = smplx_param['root_pose']
            body_pose = smplx_param['body_pose']
            jaw_pose = smplx_param['jaw_pose']
            leye_pose = smplx_param['leye_pose']
            reye_pose = smplx_param['reye_pose']
            lhand_pose = smplx_param['lhand_pose']
            rhand_pose = smplx_param['rhand_pose']
        else:
            # 当不启用姿态时，用零向量代替以获得“默认姿势”的模型输出
            root_pose = torch.zeros_like(smplx_param['root_pose'])
            body_pose = torch.zeros_like(smplx_param['body_pose'])
            jaw_pose = torch.zeros_like(smplx_param['jaw_pose'])
            leye_pose = torch.zeros_like(smplx_param['leye_pose'])
            reye_pose = torch.zeros_like(smplx_param['reye_pose'])
            lhand_pose = torch.zeros_like(smplx_param['lhand_pose'])
            rhand_pose = torch.zeros_like(smplx_param['rhand_pose'])
        
        # 表情是否启用，默认使用提供的 expr，否则替换为零向量
        if use_expr:
            expr = smplx_param['expr']
        else:
            expr = torch.zeros_like(smplx_param['expr'])

        # face/joint/locator 偏移是可选的（用于表示细节偏差），根据参数决定是否启用
        if use_face_offset:
            face_offset = smpl_x.get_face_offset(smplx_param['face_offset'])
        else:
            face_offset = None
 
        if use_joint_offset:
            joint_offset = smpl_x.get_joint_offset(smplx_param['joint_offset'])
        else:
            joint_offset = None
       
        if use_locator_offset:
            locator_offset = smpl_x.get_locator_offset(smplx_param['locator_offset'])
        else:
            locator_offset = None

        # 对 jaw_pose, leye_pose, reye_pose, and expr 进行 detach，
        # 因为这些在 flame 层中优化
        output = self.smplx_layer(
            global_orient=root_pose, body_pose=body_pose, jaw_pose=jaw_pose.detach(), 
            leye_pose=leye_pose.detach(), reye_pose=reye_pose.detach(), 
            left_hand_pose=lhand_pose, right_hand_pose=rhand_pose, 
            expression=expr.detach(), betas=smplx_param['shape'], 
            face_offset=face_offset, joint_offset=joint_offset, 
            locator_offset=locator_offset)

        # 将网格与关键点转换到以相机为中心的坐标系，先以 root 关节为基准做相对化，再加上全局平移
        mesh_cam = output.vertices
        kpt_cam = output.joints[:,smpl_x.kpt['idx'],:]
        root_cam = kpt_cam[:,smpl_x.kpt['root_idx'],:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + smplx_param['trans'][:,None,:]
        kpt_cam = kpt_cam - root_cam[:,None,:] + smplx_param['trans'][:,None,:]
        
        # 如果提供相机参数，则对关键点进行透视投影得到 2D 坐标（用于 2D 损失等）
        if cam_param is not None:
            x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
            y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
            kpt_proj = torch.stack((x,y),2)

        # 如果 root_rel 为 True，则返回相对于 root 平移的坐标（去除全局 trans）
        if root_rel:
            mesh_cam = mesh_cam - smplx_param['trans'][:,None,:]
            kpt_cam = kpt_cam - smplx_param['trans'][:,None,:]

        # 根据是否提供相机参数决定返回 3D+2D 或仅 3D
        if cam_param is not None:
            return mesh_cam, kpt_cam, root_cam, kpt_proj
        else:
             return mesh_cam, kpt_cam, root_cam

    def get_flame_coord(self, flame_param, cam_param, use_pose=True, use_expr=True):
        '''
        获取 FLAME 模型在相机坐标系下的网格与关键点坐标，并可选进行 2D 投影
        Params:
            flame_param (dict): 包含 FLAME 参数的字典
            cam_param (dict or None): 包含相机参数的字典，或 None 表示不投影
            use_pose (bool): 是否使用提供的姿态参数
            use_expr (bool): 是否使用提供的表情参数
        Returns:
            mesh_cam (tensor): (B, V, 3) 人脸网格在相机坐标系下的 3D 坐标
            kpt_cam (tensor): (B, K, 3) 关键点在相机坐标系下的 3D 坐标
            kpt_proj (tensor, optional): (B, K, 2) 关键点的 2D 投影坐标，仅当提供 cam_param 时返回
        '''
        if use_pose:
            root_pose = flame_param['root_pose']
            neck_pose = flame_param['neck_pose']
            jaw_pose = flame_param['jaw_pose']
            leye_pose = flame_param['leye_pose']
            reye_pose = flame_param['reye_pose']
        else:
            root_pose = torch.zeros_like(flame_param['root_pose'])
            neck_pose = torch.zeros_like(flame_param['neck_pose'])
            jaw_pose = torch.zeros_like(flame_param['jaw_pose'])
            leye_pose = torch.zeros_like(flame_param['leye_pose'])
            reye_pose = torch.zeros_like(flame_param['reye_pose'])
        
        # 是否使用表情分量
        if use_expr:
            expr = flame_param['expr']
        else:
            expr = torch.zeros_like(flame_param['expr'])

        # 调用 FLAME 层，得到人脸网格与关键点
        output = self.flame_layer(
            global_orient=root_pose, neck_pose=neck_pose, jaw_pose=jaw_pose, 
            leye_pose=leye_pose, reye_pose=reye_pose, expression=expr, betas=flame_param['shape']
        )

        # 将 FLAME 的输出转换为以相机为中心的 3D 坐标，并补充额外的左右耳顶点到关键点集合中
        mesh_cam = output.vertices
        kpt_cam = output.joints
        lear = mesh_cam[:,flame.lear_vertex_idx,:]
        rear = mesh_cam[:,flame.rear_vertex_idx,:]
        kpt_cam = torch.cat((kpt_cam, lear[:,None,:], rear[:,None,:]),1) # follow flame.kpt['name']
        root_cam = kpt_cam[:,flame.kpt['root_idx'],:]
        mesh_cam = mesh_cam - root_cam[:,None,:] + flame_param['trans'][:,None,:]
        kpt_cam = kpt_cam - root_cam[:,None,:] + flame_param['trans'][:,None,:]
        
        # 如提供相机参数则计算 2D 投影并返回，否则只返回 3D
        if cam_param is not None:
            # project to the 2D space
            x = kpt_cam[:,:,0] / kpt_cam[:,:,2] * cam_param['focal'][:,None,0] + cam_param['princpt'][:,None,0]
            y = kpt_cam[:,:,1] / kpt_cam[:,:,2] * cam_param['focal'][:,None,1] + cam_param['princpt'][:,None,1]
            kpt_proj = torch.stack((x,y),2) 
            return mesh_cam, kpt_cam, kpt_proj
        else:
            return mesh_cam, kpt_cam
    
    def check_face_visibility(self, face_mesh, leye, reye):
        # 根据面部几何与眼睛位置判断面部是否朝向相机，从而确定是否应该使用面部相关的 2D 损失
        # face_mesh: (B, V_face, 3) 面部顶点的 3D 坐标
        # leye, reye: (B, 3) 左/右眼关键点坐标
        '''
        判断面部是否朝向相机
        Params:
            face_mesh: (B, V_face, 3) 面部顶点的 3D 坐标
            leye, reye: (B, 3) 左/右眼关键点坐标
        Returns:
            face_valid: (B,) 布尔张量，True 表示面部朝向相机
        Notes:
            计算面部中心指向眼睛的向量与面部中心指向相机的向量的夹角
            角度过大（内积小于 cos(3*pi/4)）则认为面部未朝向相机
        '''
        center = face_mesh.mean(1)
        eye = (leye + reye)/2.

        # 计算从面部中心指向眼睛与面部中心指向相机方向的向量（仅保留 x,z 维，忽略 y）
        eye_vec = eye - center
        cam_vec = center - 0

        eye_vec = F.normalize(torch.stack((eye_vec[:,0], eye_vec[:,2]),1), p=2, dim=1)
        cam_vec = F.normalize(torch.stack((cam_vec[:,0], cam_vec[:,2]),1), p=2, dim=1)

        # 内积用于度量朝向一致性；阈值使用 cos(3*pi/4)（可以视为较宽松的朝向判定）
        dot_prod = torch.sum(eye_vec * cam_vec, 1)
        face_valid = dot_prod < math.cos(math.pi/4*3)
        return face_valid

    def get_smplx_full_pose(self, smplx_param):
        # 将各部分姿态拼接成完整的关节姿态向量，顺序遵循 smpl_x.joint['name']
        pose = torch.cat((smplx_param['root_pose'][:,None,:], 
                          smplx_param['body_pose'], 
                          smplx_param['jaw_pose'][:,None,:], 
                          smplx_param['leye_pose'][:,None,:], 
                          smplx_param['reye_pose'][:,None,:], 
                          smplx_param['lhand_pose'], 
                          smplx_param['rhand_pose']),1) # follow smpl_x.joint['name']
        return pose
 
    def get_flame_full_pose(self, flame_param):
        # 拼接 FLAME 的关节姿态表示（不包含根节点），用于 pose 损失计算
        pose = torch.cat((flame_param['neck_pose'][:,None,:], 
                          flame_param['jaw_pose'][:,None,:], 
                          flame_param['leye_pose'][:,None,:], 
                          flame_param['reye_pose'][:,None,:]),1) # follow flame.joint['name'] without the root joint
        return pose
   
    def forward(self, smplx_inputs, flame_inputs, data, return_output):
        '''
        
        '''
        # 预处理输入参数：将 6D 旋转转换为 axis-angle，并整理字段结构
        smplx_inputs = self.process_input_smplx_param(smplx_inputs) 
        flame_inputs = self.process_input_flame_param(flame_inputs) 
       
        # 使用当前可优化参数计算 SMPL-X / FLAME 的 3D 网格与 2D 投影（若提供 cam_param）
        smplx_mesh_cam, smplx_kpt_cam, smplx_root_cam, smplx_kpt_proj = \
            self.get_smplx_coord(smplx_inputs, data['cam_param_proj'])
        # 不启用 face_offset 时的对照结果（用于某些损失项比较）
        smplx_mesh_cam_wo_fo, smplx_kpt_cam_wo_fo, \
            smplx_root_cam_wo_fo, smplx_kpt_proj_wo_fo = \
                self.get_smplx_coord(smplx_inputs, data['cam_param_proj'], use_face_offset=False)
        flame_mesh_cam, flame_kpt_cam, flame_kpt_proj = self.get_flame_coord(flame_inputs, data['cam_param_proj'])
        
        # 计算“零姿态/零表情”下的网格（用于正则化/结构约束等），不投影到 2D，仅获取 3D
        smplx_mesh_wo_pose_wo_expr, _, _ = \
            self.get_smplx_coord(smplx_inputs, cam_param=None, use_pose=False, 
                                 use_expr=False, use_locator_offset=False, root_rel=True)
        flame_mesh_wo_pose_wo_expr, _ = \
            self.get_flame_coord(flame_inputs, cam_param=None, use_pose=False, use_expr=False)

        # 使用初始参数（通常为从数据加载的初始化 SMPL-X 参数）计算初始网格以判定可见性
        with torch.no_grad():
            # 复制当前计算得到的初始参数到 data['smplx_param']，并 detach
            data['smplx_param']['shape'] = smplx_inputs['shape'].clone().detach()
            data['smplx_param']['expr'] = smplx_inputs['expr'].clone().detach()
            data['smplx_param']['trans'] = smplx_inputs['trans'].clone().detach()
            data['smplx_param']['joint_offset'] = smplx_inputs['joint_offset'].clone().detach()
            data['smplx_param']['locator_offset'] = smplx_inputs['locator_offset'].clone().detach()
            smplx_mesh_cam_init, smplx_kpt_cam_init, _, _ = \
                self.get_smplx_coord(data['smplx_param'], data['cam_param_proj'], use_face_offset=False)

            # 基于初始网格判断哪些帧/样本的面部是可见的（用于决定是否计算某些 2D 损失）
            face_valid = self.check_face_visibility(smplx_mesh_cam_init[:,smpl_x.face_vertex_idx,:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('L_Eye'),:], smplx_kpt_cam_init[:,smpl_x.kpt['name'].index('R_Eye'),:])
            face_valid = face_valid * data['flame_valid']

        # 下面构造各类损失项并聚合到字典中返回
        loss = {}
        # 初始化用于 2D 关键点损失的权重张量
        weight = torch.ones_like(smplx_kpt_proj)
        if not cfg.warmup:
            # 在非 warmup 阶段，默认先屏蔽所有面部关键点（由其它损失/约束替代），再根据 face_valid 恢复可见帧
            weight[:,[i for i in range(smpl_x.kpt['num']) if 'Face' in smpl_x.kpt['name'][i]],:] = 0
            weight[face_valid,:,:] = 1 # do not use 2D loss if face is not visible
        # 2D 关键点投影损失：SMPL-X 的投影 vs 图像关键点
        loss['smplx_kpt_proj'] = self.coord_loss(smplx_kpt_proj, data['kpt_img'], data['kpt_valid'], smplx_kpt_cam.detach()) * weight
        loss['smplx_kpt_proj_wo_fo'] = self.coord_loss(smplx_kpt_proj_wo_fo, data['kpt_img'], data['kpt_valid'], smplx_kpt_cam.detach()) * weight
        # FLAME 的 2D 人脸关键点损失（只对人脸部分计算）
        loss['flame_kpt_proj'] = torch.abs(flame_kpt_proj - data['kpt_img'][:,smpl_x.kpt['part_idx']['face'],:]) * data['kpt_valid'][:,smpl_x.kpt['part_idx']['face'],:] * weight[:,smpl_x.kpt['part_idx']['face'],:]
        # 若处于 warmup 阶段，使用较简单的点到点误差作为初始化引导
        if cfg.warmup:
            loss['flame_to_smplx_v2v'] = torch.abs(flame_mesh_cam - smplx_mesh_cam[:,smpl_x.face_vertex_idx,:].detach())
        else:
            # 非 warmup 阶段加入更多的正则化与结构一致性约束
            loss['smplx_shape_reg'] = smplx_inputs['shape'] ** 2 * 0.01
            loss['smplx_mesh'] = torch.abs((smplx_mesh_cam_wo_fo - smplx_kpt_cam_wo_fo[:,smpl_x.kpt['root_idx'],None,:]) - \
                                            (smplx_mesh_cam_init - smplx_kpt_cam_init[:,smpl_x.kpt['root_idx'],None,:])) * 0.1 
            # SMPL-X 相关的 pose 约束
            smplx_input_pose = self.get_smplx_full_pose(smplx_inputs)
            smplx_init_pose = self.get_smplx_full_pose(data['smplx_param'])
            loss['smplx_pose'] = self.pose_loss(smplx_input_pose, smplx_init_pose) * 0.1
            # 对躯干/颈部的 pose 做 L2 正则，避免前倾头等不良姿态
            loss['smplx_pose_reg'] = torch.stack(
                [smplx_input_pose[:,i,0] for i in range(smpl_x.joint['num']) 
                 if smpl_x.joint['name'][i] in ['Spine_1', 'Spine_2', 'Spine_3', 'Neck', 'Head']],
                 1) ** 2  # prevent forward head posture

            # FLAME 相关的 pose/shape/expr 约束
            flame_input_pose = self.get_flame_full_pose(flame_inputs)
            flame_init_pose = self.get_flame_full_pose(data['flame_param'])
            loss['flame_pose'] = self.pose_loss(flame_input_pose, flame_init_pose) * 0.1
            loss['flame_shape'] = torch.abs(flame_inputs['shape'] - data['flame_param']['shape']) * 0.1
            loss['flame_expr'] = torch.abs(flame_inputs['expr'] - data['flame_param']['expr']) * 0.1

            # 一系列用于对齐 SMPL-X 与 FLAME 的形状一致性与网格正则化的损失
            is_not_neck = torch.ones((1,flame.vertex_num,1)).float().cuda()
            is_not_neck[:,flame.layer.lbs_weights.argmax(1)==flame.joint['root_idx'],:] = 0
            loss['smplx_to_flame_v2v_wo_pose_expr'] = torch.abs(\
                    (smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:] - smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:].mean(1)[:,None,:]) - \
                    (flame_mesh_wo_pose_wo_expr - flame_mesh_wo_pose_wo_expr.mean(1)[:,None,:]).detach()) * is_not_neck * 10
            loss['smplx_to_flame_lap'] = self.lap_reg(smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:], flame_mesh_wo_pose_wo_expr.detach()) * is_not_neck * 100000
            loss['smplx_to_flame_edge_length'] = self.edge_length_loss(smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:], flame_mesh_wo_pose_wo_expr.detach(), is_not_neck)

            # 对偏移项（face/joint/locator）施加正则与对称性约束，重点保护颈部区域
            is_neck = torch.zeros((1,flame.vertex_num,1)).float().cuda()
            is_neck[:,flame.layer.lbs_weights.argmax(1)==flame.joint['root_idx'],:] = 1
            loss['face_offset_reg'] = smplx_inputs['face_offset'] ** 2 * is_neck * 1000
            weight = torch.ones((1,smpl_x.joint['num'],1)).float().cuda()
            if not cfg.hand_joint_offset:
                weight[:,smpl_x.joint['part_idx']['lhand'],:] = 10
                weight[:,smpl_x.joint['part_idx']['rhand'],:] = 10
            loss['joint_offset_reg'] = smplx_inputs['joint_offset'] ** 2 * 100 * weight
            loss['locator_offset_reg'] = smplx_inputs['locator_offset'] ** 2
            loss['face_offset_sym_reg'] = self.face_offset_sym_reg(smplx_inputs['face_offset'])
            loss['joint_offset_sym_reg'] = self.joint_offset_sym_reg(smplx_inputs['joint_offset'])
            loss['locator_offset_sym_reg'] = self.joint_offset_sym_reg(smplx_inputs['locator_offset'])
        
        # 如果不需要可视化/输出细节，仅返回 loss
        if not return_output:
            return loss, None
        else:
            # - smplx_mesh_cam_wo_jo: 不应用 joint_offset 的 SMPL-X 网格
            smplx_mesh_cam_wo_jo, _, _ = self.get_smplx_coord(
                smplx_inputs, cam_param=None, use_joint_offset=False
            )
            # - smplx_mesh_wo_pose_wo_expr_wo_fo: 无 pose/expr/face_offset 的参考网格（root_rel=True）
            smplx_mesh_wo_pose_wo_expr_wo_fo, _, _ = self.get_smplx_coord(
                smplx_inputs, cam_param=None, use_pose=False, use_expr=False, 
                use_face_offset=False, use_locator_offset=False, root_rel=True
            )

            # 为了在同一坐标系下对齐展示，将 FLAME 网格整体平移到与 SMPL-X 的人脸中心对齐
            offset = -flame_mesh_wo_pose_wo_expr.mean(1) + \
                smplx_mesh_wo_pose_wo_expr[:,smpl_x.face_vertex_idx,:].mean(1)
            flame_mesh_wo_pose_wo_expr = flame_mesh_wo_pose_wo_expr + offset[:,None,:]
            
            # 组装输出字典，包含训练时/可视化时需要的网格与位移信息
            out = {}
            out['smplx_mesh_cam'] = smplx_mesh_cam
            out['smplx_mesh_cam_wo_jo'] = smplx_mesh_cam_wo_jo
            out['smplx_mesh_cam_wo_fo'] = smplx_mesh_cam_wo_fo
            out['smplx_trans'] = smplx_inputs['trans'] - smplx_root_cam
            out['flame_mesh_cam'] = flame_mesh_cam
            out['smplx_mesh_wo_pose_wo_expr'] = smplx_mesh_wo_pose_wo_expr
            out['smplx_mesh_wo_pose_wo_expr_wo_fo'] = smplx_mesh_wo_pose_wo_expr_wo_fo
            out['flame_mesh_wo_pose_wo_expr'] = flame_mesh_wo_pose_wo_expr
            return loss, out
 
def get_model():
    model = Model()
    return model
