import os
import os.path as osp
from glob import glob
import sys
import argparse
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.logger import colorlogger

logger = colorlogger()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, dest='root_path')
    parser.add_argument('--use_colmap', dest='use_colmap', action='store_true')
    args = parser.parse_args()
    assert args.root_path, "Please set root_path."
    return args

# get path
args = parse_args()
root_path = args.root_path
if root_path[-1] == '/':
    subject_id = root_path.split('/')[-2]
else:
    subject_id = root_path.split('/')[-1]

# remove unnecessary frames
# 删除不需要的帧，只保留 frame_list_all.txt 中列出的那些帧。
with open(osp.join(root_path, 'frame_list_all.txt')) as f:
    frame_idx_list = [int(x) for x in f.readlines()]
img_path_list = glob(osp.join(root_path, 'frames', '*.png'))
for img_path in img_path_list:
    frame_idx = int(img_path.split('/')[-1][:-4])
    if frame_idx not in frame_idx_list:
        cmd = 'rm ' + img_path
        result = os.system(cmd)
        if (result != 0):
            print('something bad happened when removing unnecessary frames. terminate the script.')
            sys.exit()

# make camera parameters
# 调用 COLMAP 或生成虚拟相机参数
if args.use_colmap:
    logger.info('Running COLMAP to get camera parameters...')
    os.chdir('./COLMAP')
    cmd = 'python run_colmap.py --root_path ' + root_path
    logger.debug(cmd)
    result = os.system(cmd)
    if (result != 0):
        logger.error('something bad happened when running COLMAP to get camera parameters. terminate the script.')
        sys.exit()
    logger.info(f'Camera parameters are generated at [{osp.join(root_path, "sparse")}]')
else:
    logger.info('Making virtual camera parameters...')
    cmd = 'python make_virtual_cam_params.py --root_path ' + root_path
    logger.debug(cmd)
    result = os.system(cmd)
    if (result != 0):
        logger.error('something bad happened when making the virtual camera parameters. terminate the script.')
        sys.exit()
    logger.info(f'Virtual camera parameters are generate at [{osp.join(root_path, "cam_params")}]')

# DECA (get initial FLAME parameters)
# 调用 DECA 获取初始 FLAME 面部参数
logger.info('Running DECA to get initial FLAME parameters...')
os.chdir('./DECA')
cmd = 'python run_deca.py --root_path ' + root_path
print(cmd)
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when running DECA. terminate the script.')
    sys.exit()
os.chdir('..')
logger.info(f'FLAME parameters are generated at [{osp.join(root_path, "flame_init")}]')


# Hand4Whole (get initial SMPLX parameters)
# 调用 Hand4Whole 获取初始 SMPLX 全身参数
logger.info('Running Hand4Whole to get initial SMPLX parameters...')
os.chdir('./Hand4Whole_RELEASE/demo')
cmd = 'python run_hand4whole.py --gpu 0 --root_path ' + root_path
logger.debug(cmd)
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when running Hand4Whole. terminate the script.')
    sys.exit()
os.chdir('../../')
logger.info(f'SMPLX parameters are generated at [{osp.join(root_path, "smplx_init")}]')
logger.info(f'Visualization video is saved at [{osp.join(root_path, "smplx_init.mp4")}]')


# mmpose (get 2D whole-body keypoints)
# 调用 mmpose 获取 2D 全身关键点
logger.info('Running mmpose to get 2D whole-body keypoints...')
os.chdir('./mmpose')
cmd = 'python run_mmpose.py --root_path ' + root_path
logger.debug(cmd)
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when running mmpose. terminate the script.')
    sys.exit()
os.chdir('..')
logger.info(f'2D whole-body keypoints are generated at [{osp.join(root_path, "keypoints_whole_body")}]')


# fit SMPLX
logger.info('Fitting SMPLX to get optimized SMPLX parameters...')
os.chdir('../main')
cmd = 'python fit.py --subject_id ' + subject_id
logger.debug(cmd)
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when fitting. terminate the script.')
    sys.exit()
os.chdir('../tools')
cmd = 'rm -rf ' + osp.join(root_path, 'smplx_optimized')
logger.debug(cmd)
result = os.system(cmd)
cmd = 'mkdir ' + osp.join(root_path, 'smplx_optimized')
logger.debug(cmd)
result = os.system(cmd)
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, '*') + ' ' + osp.join(root_path, '.')
logger.debug(cmd)
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when moving the fitted files to root_path. terminate the script.')
    sys.exit()
logger.info(f'Optimized SMPLX parameters are moved to [{osp.join(root_path, "smplx_optimized")}]')


# unwrap textures of FLAME
os.chdir('../main')
logger.info('Unwrapping face images to FLAME UV texture...')
cmd = 'python unwrap.py --subject_id ' + subject_id
logger.debug(cmd)
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when unwrapping the face images to FLAME UV texture. terminate the script.')
    sys.exit()
os.chdir('../tools')
cmd = 'mv ' + osp.join('..', 'output', 'result', subject_id, 'unwrapped_textures', '*') + ' ' + osp.join(root_path, 'smplx_optimized', '.')
logger.info(f'Unwrapped FLAME UV texture is moved to [{osp.join(root_path, "smplx_optimized")}]')
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when moving the unwrapped FLAME UV texture to root_path. terminate the script.')
    sys.exit()

# smooth SMPLX
logger.info('Smoothing SMPLX parameters over frames...')
cmd = 'python smooth_smplx_params.py --root_path ' + root_path
logger.debug(cmd)
result = os.system(cmd)
if (result != 0):
    logger.error('something bad happened when smoothing smplx parameters. terminate the script.')
    sys.exit()

