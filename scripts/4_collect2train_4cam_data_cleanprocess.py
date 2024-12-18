import time
import h5py
import os
from constants import PUPPET_GRIPPER_POSITION_NORMALIZE_FN, SIM_TASK_CONFIGS, DT
import glob
import cv2
import pickle
import numpy as np
from scripts.function_util import save_videos, mk_dir
from pathlib import Path
import tyro
from dataclasses import dataclass
import gc  # 添加gc模块用于内存清理

#
# """
# For each timestep:
# observations
# - images
#     - each_cam_name     (480, 640, 3) 'uint8'
# - qpos                  (14,)         'float64'
# - qvel                  (14,)         'float64'
#
# action                  (14,)         'float64'
# """
#

@dataclass
class Args:
    dataset_name: str ="collect_1119_100_yellow_plasticbag_M_onlygrabbag"
    task_name: str ="collect_1119_100_yellow_plasticbag_M_onlygrabbag_task"
    MIRROR_STATE_MULTIPLY: list = (1, 1, 1, 1, 1, 1, 1)
    MIRROR_BASE_MULTIPLY: tuple = (1, 1)
    save_video: bool = True  # 添加是否保存视频的选项


def deal_data(pos_list, top_list, left_list, right_list, bottom_list):
    print('aga测试：进入deal_data内部')
    if len(pos_list) < len(top_list):
        for i in range(len(top_list)):
            file_name = top_list[i].split("/")[-1].split(".")[0] + ".pkl"
            if not os.path.exists(os.path.dirname(pos_list[0])+f"/{file_name}"):
                print(top_list[i])
                os.remove(top_list[i])
                os.remove(left_list[i])
                os.remove(right_list[i])
                os.remove(bottom_list[i])
                top_list.remove(top_list[i])
                left_list.remove(left_list[i])
                right_list.remove(right_list[i])
                bottom_list.remove(bottom_list[i])
    elif len(pos_list) > len(top_list):
        for i in range(len(pos_list)):
            file_name = pos_list[i].split("/")[-1].split(".")[0] + ".jpg"
            if not os.path.exists(os.path.dirname(pos_list[0])+f"/{file_name}"):
                print(pos_list[i])
                os.remove(pos_list[i])
                pos_list.remove(pos_list[i])
    return pos_list, top_list, left_list, right_list, bottom_list


def load_data(args, one_dataset_dir):
    print('aga测试：进入load_data内部')
    camera_names = SIM_TASK_CONFIGS[args.task_name]['camera_names']
    print(camera_names)
    print('aga测试：进入获得文件名到XXX_list')
    data_pose_list = glob.glob(one_dataset_dir + 'observation/*.pkl')
    images_top_list = glob.glob(one_dataset_dir + 'topImg/*.jpg')
    images_left_list = glob.glob(one_dataset_dir + 'leftImg/*.jpg')
    images_right_list = glob.glob(one_dataset_dir + 'rightImg/*.jpg')
    images_bottom_list = glob.glob(one_dataset_dir + 'bottomImg/*.jpg')
    print('aga测试：进入排序')
    data_pose_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_top_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_left_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_right_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    images_bottom_list.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))
    print('aga测试：进入load_data中的deal_data')
    data_pose_list, images_top_list, images_left_list, images_right_list, images_bottom_list = (
        deal_data(data_pose_list, images_top_list, images_left_list, images_right_list, images_bottom_list))
    
    print('aga测试：创建初始化列表')
    is_sim = False
    qpos = []
    qvel = []
    action = []
    base_action = None
    image_dict = dict()
    image_li = [[], [], [], []]
    for cam_name in camera_names:
        image_dict[f'{cam_name}'] = []
    for i in range(len(data_pose_list)):
        with open(data_pose_list[i], "rb") as f:
            data_single = pickle.load(f)
            qpos.append(data_single['joint_positions'])
            qvel.append(data_single['joint_velocities'])
            action.append(data_single['control'])
            image_top = cv2.imread(images_top_list[i])
            image_left = cv2.imread(images_left_list[i])
            image_right = cv2.imread(images_right_list[i])
            image_bottom = cv2.imread(images_bottom_list[i])
            # [新增] 检查图像加载情况，只在尺寸不是(480,640,3)时打印
            expected_shape = (480, 640, 3)
            if (image_top is not None and image_top.shape != expected_shape or 
                image_left is not None and image_left.shape != expected_shape or 
                image_right is not None and image_right.shape != expected_shape or 
                image_bottom is not None and image_bottom.shape != expected_shape):
                print(f"\n=== 第{i}轮图片尺寸不是(480,640,3) ===")
                if image_top is not None and image_top.shape != expected_shape:
                    print(f"Top image shape: {image_top.shape}")
                if image_left is not None and image_left.shape != expected_shape:
                    print(f"Left image shape: {image_left.shape}")
                if image_right is not None and image_right.shape != expected_shape:
                    print(f"Right image shape: {image_right.shape}")
                if image_bottom is not None and image_bottom.shape != expected_shape:
                    print(f"Bottom image shape: {image_bottom.shape}")
            
            image_li[0].append(image_top)
            image_li[1].append(image_left)
            image_li[2].append(image_right)
            image_li[3].append(image_bottom)
    print('aga测试：进入将image_li转换为image_dict')
    image_dict['top'] = np.array(image_li[0])
    print('aga测试：top转存已完成')
    image_dict['left_wrist'] = np.array(image_li[1])
    print('aga测试：left_wrist转存已完成')
    image_dict['right_wrist'] = np.array(image_li[2])
    print('aga测试：right_wrist转存已完成')
    #bottom转存前读取当前内存占用
    print('aga测试：进入bottom转存前内存占用：', gc.get_stats())
    #bottom转存前打印image_li[2]和image_li[3]的形状
    print('aga测试：被转存到right_wrist的image_li[2]的形状：', np.array(image_li[2]).shape)
    print('aga测试：被转存到bottom的image_li[3]的形状：', np.array(image_li[3]).shape)
    image_dict['bottom'] = np.array(image_li[3])
    print('aga测试：bottom转存已完成')
    
    return np.array(qpos), np.array(qvel), np.array(action), base_action, image_dict, is_sim


def main(args):
    root_dir = str("/home/asdfminer/YidongYingPan/XtrainerCollectedData/datasets/")
    dataset_dir = root_dir + "/" + args.dataset_name + "/collect_data/"
    mk_dir(dataset_dir)
    output_video_dir = root_dir + "/" + args.dataset_name + "/output_videos/"
    mk_dir(output_video_dir)
    output_train_data = root_dir + "/" + args.dataset_name + "/train_data/"
    mk_dir(output_train_data)

    all_data_dir = os.listdir(dataset_dir)
    all_data_dir.sort(key=lambda x: int(x))
    
    # 检查已处理的文件
    print('aga测试：进入检查已处理文件')
    processed_files = set()
    for f in os.listdir(output_train_data):
        if f.endswith('.hdf5'):
            idx = int(f.split('_')[-1].split('.')[0])
            processed_files.add(idx)
    
    # 如果存在已处理的文件，删除最后一个处理的文件
    if processed_files:
        last_processed = max(processed_files)
        last_file = os.path.join(output_train_data, f'episode_init_{last_processed}.hdf5')
        if os.path.exists(last_file):
            os.remove(last_file)
            print(f"删除上次处理的最后一个文件: {last_file}")
            processed_files.remove(last_processed)

    MIRROR_STATE_MULTIPLY = np.array(args.MIRROR_STATE_MULTIPLY)

    for idx in range(len(all_data_dir)):
        # 跳过已处理的文件
        if idx in processed_files:
            print(f"跳过已处理的数据: {idx}")
            continue
            
        print("正在处理: ", idx)
        one_data_dir = dataset_dir+all_data_dir[idx]+"/"
        print(one_data_dir)
        
        # 检查输出文件是否已存在
        output_file = os.path.join(output_train_data, f'episode_init_{idx}.hdf5')
        if os.path.exists(output_file):
            print(f"文件已存在，跳过: {output_file}")
            continue
        
        print('aga测试：进入加载数据')
        qpos, qvel, action, base_action, image_dict, is_sim = load_data(args, one_data_dir)
        print('aga测试：进入使用mirror处理数据')
        qpos = np.concatenate([qpos[:, :7] * MIRROR_STATE_MULTIPLY, qpos[:, 7:] * MIRROR_STATE_MULTIPLY], axis=1)
        qvel = np.concatenate([qvel[:, :7] * MIRROR_STATE_MULTIPLY, qvel[:, 7:] * MIRROR_STATE_MULTIPLY], axis=1)
        action = np.concatenate([action[:, :7] * MIRROR_STATE_MULTIPLY, action[:, 7:] * MIRROR_STATE_MULTIPLY], axis=1)
        if base_action is not None:
            base_action = base_action * args.MIRROR_BASE_MULTIPLY

        if 'left_wrist' in image_dict.keys():
            image_dict['left_wrist'], image_dict['right_wrist'] = \
                image_dict['left_wrist'], image_dict['right_wrist']
        elif 'cam_left_wrist' in image_dict.keys():
            image_dict['cam_left_wrist'], image_dict['cam_right_wrist'] = \
                image_dict['cam_left_wrist'][:, :, ::-1], image_dict['cam_right_wrist'][:, :, ::-1]
        else:
            raise Exception('No left_wrist or cam_left_wrist in image_dict')

        if 'top' in image_dict.keys():
            image_dict['top'] = image_dict['top']
        elif 'cam_high' in image_dict.keys():
            image_dict['cam_high'] = image_dict['cam_high'][:, :, ::-1]
        else:
            raise Exception('No top or cam_high in image_dict')

        print('aga测试：进入保存数据')
        # saving
        data_dict = {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
            '/base_action': base_action,
        } if base_action is not None else {
            '/observations/qpos': qpos,
            '/observations/qvel': qvel,
            '/action': action,
        }
        for cam_name in image_dict.keys():
            data_dict[f'/observations/images/{cam_name}'] = image_dict[cam_name]
        max_timesteps = len(qpos)
        
        print('aga测试：进入压缩数据')
        COMPRESS = True

        if COMPRESS:
            # JPEG compression
            t0 = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # tried as low as 20, seems fine
            compressed_len = []
            for cam_name in image_dict.keys():
                image_list = data_dict[f'/observations/images/{cam_name}']
                compressed_list = []
                compressed_len.append([])
                for image in image_list:
                    result, encoded_image = cv2.imencode('.jpg', image,
                                                         encode_param)  # 0.02 sec # cv2.imdecode(encoded_image, 1)
                    compressed_list.append(encoded_image)
                    compressed_len[-1].append(len(encoded_image))
                data_dict[f'/observations/images/{cam_name}'] = compressed_list
            print(f'compression: {time.time() - t0:.2f}s')

            
            # pad so it has same length
            print('aga测试：进入填充数据')
            t0 = time.time()
            compressed_len = np.array(compressed_len)
            padded_size = compressed_len.max()
            for cam_name in image_dict.keys():
                compressed_image_list = data_dict[f'/observations/images/{cam_name}']
                padded_compressed_image_list = []
                for compressed_image in compressed_image_list:
                    padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                    image_len = len(compressed_image)
                    padded_compressed_image[:image_len] = compressed_image
                    padded_compressed_image_list.append(padded_compressed_image)
                data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
            print(f'padding: {time.time() - t0:.2f}s')

        # HDF5
        print('aga测试：进入保存hdf5')
        t0 = time.time()
        dataset_path = os.path.join(output_train_data, f'episode_init_{idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = is_sim
            root.attrs['compress'] = COMPRESS
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in image_dict.keys():
                if COMPRESS:
                    _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                             chunks=(1, padded_size), )
                else:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))
            if base_action is not None:
                base_action = root.create_dataset('base_action', (max_timesteps, 2))

            for name, array in data_dict.items():
                root[name][...] = array

            if COMPRESS:
                _ = root.create_dataset('compress_len', (len(image_dict.keys()), max_timesteps))
                root['/compress_len'][...] = compressed_len

        print(f'Saving {dataset_path}: {time.time() - t0:.1f} secs\n')

        # 根据参数决定是否保存视频
        if args.save_video:
            print('aga测试：进入保存视频')
            save_videos(image_dict, DT, video_path=os.path.join(output_video_dir + f'{all_data_dir[idx]}_video.mp4'))
            print(f"已保存视频: {all_data_dir[idx]}_video.mp4")

        # 处理完成后清理内存
        print('aga测试：进入清理内存')
        gc.collect()
        
        # 清理大型变量
        del qpos, qvel, action, base_action, image_dict
        gc.collect()

        print(f"完成处理 {idx}, 内存已清理")


if __name__ == "__main__":
    main(tyro.cli(Args))