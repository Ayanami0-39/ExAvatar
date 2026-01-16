# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

from . import detectors

def video2sequence(video_path, sample_step=10):
    """
    将视频转换为图像序列
    Params:
        video_path: 视频文件路径
        sample_step: 采样间隔
    Returns:
        imagepath_list: 提取出的图像帧路径列表
    """
    # 获取视频所在的文件夹路径，并以视频名创建子文件夹
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    
    # 获取视频文件名（不含扩展名）
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    
    # 打开视频文件
    vidcap = cv2.VideoCapture(video_path)
    # 读取第一帧
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    
    # 循环读取每一帧直到结束
    while success:
        # if count%sample_step == 0:  # 这里 sample_step 被注释掉了，表示当前提取所有帧
        # 构建当前帧的保存路径，格式为：视频名_frameXXXX.jpg
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:04d}.jpg')
        cv2.imwrite(imagepath, image)     # 保存帧为 JPEG 文件
        
        # 继续读取下一帧
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
        
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list

class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', sample_step=10):
        '''
        加载输入路径下的图像数据集，支持文件夹、图片列表、单张图片或视频路径。
        进行人脸检测和裁剪，返回处理后的图像数据。
        
        Args:
            testpath: 输入路径，可以是文件夹、图片路径列表、单张图片路径或视频路径
            iscrop: 是否裁剪人脸区域
            crop_size: 裁剪后 resize 的目标大小 (默认 224)
            scale: 裁剪框的扩充比例 (默认 1.25)
            face_detector: 使用的人脸检测器 ('fan' 或 'mtcnn')，默认为 'fan'
            sample_step: 视频采样间隔
        '''
        # 处理不同类型的输入路径
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            # 如果是文件夹，获取其中所有的图片文件 (jpg, png, bmp)
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            # 如果是单张图片
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            # 如果是视频文件，先转换为图像序列
            self.imagepath_list = video2sequence(testpath, sample_step)
        else:
            print(f'please check the test path: {testpath}')
            exit()
            
        # 排序文件列表，确保处理顺序一致
        # print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        
        # 初始化参数
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        
        # 初始化人脸检测器
        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
        # elif face_detector == 'mtcnn':
        #     self.face_detector = detectors.MTCNN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' 
        将检测到的边界框或关键点范围转换为中心点和尺寸，便于后续裁剪
        
        Args:
            left, right, top, bottom: 边界框或关键点范围的坐标
            type: 输入类型 ('bbox' 或 'kpt68')
                  'bbox': 来自人脸检测器的边界框
                  'kpt68': 来自 68 个关键点的范围
        
        Returns:
            old_size: 裁剪区域的大小的一半 (类似于半径)
            center: 裁剪区域的中心坐标 [x, y]
        '''
        # 这里的 bbox 和 landmarks 的中心点定义略有不同
        if type=='kpt68':
            # 基于关键点计算：size 稍微放大 1.1 倍
            old_size = (right - left + bottom - top)/2*1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        elif type=='bbox':
            # 基于检测框计算
            old_size = (right - left + bottom - top)/2
            # 中心点稍微向下偏移，因为人脸检测框通常比较紧凑，向下偏移可以包含更多颈部/下巴
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0  + old_size*0.12])
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        # 获取当前图像的路径和名称
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        
        # 读取图像并将其转换为 numpy 数组
        image = np.array(imread(imagepath))
        # 确保图像为 3 通道 (RGB)
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:,:,:3]

        is_valid = True
        h, w, _ = image.shape
        
        # 如果需要进行人脸裁剪 (True by default)
        if self.iscrop:
            # 检查是否有现成的关键点文件 (.mat 或 .txt)
            # 例如 AFLW2000 数据集提供了 .mat 格式的关键点
            kpt_matpath = os.path.splitext(imagepath)[0]+'.mat'
            kpt_txtpath = os.path.splitext(imagepath)[0]+'.txt'
            
            if os.path.exists(kpt_matpath):
                # 从 .mat 文件加载 68 个关键点
                kpt = scipy.io.loadmat(kpt_matpath)['pt3d_68'].T        
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            elif os.path.exists(kpt_txtpath):
                # 从 .txt 文件加载关键点
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
            else:
                # 如果没有关键点文件，使用人脸检测器 (FAN) 检测人脸
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    # 如果未检测到人脸，使用整张图像
                    # print('no face detected! run original image')
                    left = 0; right = h-1; top=0; bottom=w-1
                    is_valid = False
                    old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                else:
                    # 获取检测框坐标
                    left = bbox[0]; right=bbox[2]
                    top = bbox[1]; bottom=bbox[3]
                    old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
            
            # 计算裁剪框的大小（应用扩缩放比例 self.scale）
            size = int(old_size*self.scale)
            # 定义源图像中的三个关键点：左上角、左下角、右上角 (用于仿射变换)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            # 如果不裁剪，使用整张图像，源点为图像的三个角
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        # 定义目标图像中的三个对应点 (用于 resize 到 self.resolution_inp)
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        # 估计相似变换矩阵 (Similarity Transform)
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # 归一化图像至 [0, 1]
        image = image/255.

        # 使用 estimated transform 对图像进行 warp (裁剪 + 缩放 + 旋转)
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        # 转换为 PyTorch 格式 (C, H, W)
        dst_image = dst_image.transpose(2,0,1)
        
        # 返回处理后的数据字典
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': torch.tensor(tform.params).float(), # 变换矩阵
                'original_image': torch.tensor(image.transpose(2,0,1)).float(), # 原始图像 (未 crop)
                'is_valid': is_valid # 是否检测到人脸
                }
