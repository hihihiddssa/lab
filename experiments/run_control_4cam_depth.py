'''
我使用的intel的realsense相机。
我的程序:runcontrolAGA graduationproject是采集信息/collect2train 4cam aga graduationproject是将采集到的数据处理。
请你充分阅读之后，将深度信息加入采集数据之中，确保最后可以和其他数据一起处理到hdf5文件中。
'''

import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.format_obs import save_frame
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
import datetime
from pathlib import Path
import matplotlib.pyplot as plt



# 定义命令行参数的数据类
'''
这里装饰器等价于：
class Args:
    def __init__(self, 
                 robot_port: int = 6001, 
                 hostname: str = "127.0.0.1", 
                 show_img: bool = True, 
                 save_data_path: str = "/home/asdfminer/YidongYingPan/XtrainerCollectedData/datasets/", 
                 project_name: str = "AGA_graduation_project_1022"):
        self.robot_port = robot_port
        self.hostname = hostname
        self.show_img = show_img
        self.save_data_path = save_data_path
        self.project_name = project_name

    def __repr__(self):
        return (f"Args(robot_port={self.robot_port}, hostname={self.hostname}, "
                f"show_img={self.show_img}, save_data_path={self.save_data_path}, "
                f"project_name={self.project_name})")
'''
@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True
    save_data_path = str("/home/asdfminer/YidongYingPan/XtrainerCollectedData") + "/datasets/"
    project_name = "AGA_graduation_project_1022"
   

# 定义按钮状态的全局变量
# what_to_do: [lock or not, servo or not, record or not]
# 0: lock, 1: unlock
# 0: stop servo, 1: servo
# 0: stop recording, 1: recording
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])

# 实时监控按钮状态的线程函数
def button_monitor_realtime(agent):
    last_keys_status = np.array(([0, 0], [0, 0]))  # 上一次的按键状态
    start_press_status = np.array(([0, 0], [0, 0]))  # 按键按下的初始状态
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))  # 按键按下的计数

    while 1:
        now_keys = agent.get_keys()  # 获取当前按键状态
        dev_keys = now_keys - last_keys_status  # 计算按键状态的变化
        # 处理按钮A的按下和释放事件
        for i in range(2):
            if dev_keys[i, 0] == -1:  # 按钮A按下
                tic = time.time()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # 按钮A释放
                start_press_status[i, 0] = 0
                toc = time.time()
                if toc - tic < 0.5:  # 短按事件
                    keys_press_count[i, 0] += 1
                    if keys_press_count[i, 0] % 2 == 1:
                        what_to_do[i, 0] = 1  # 解锁
                        print("ButtonA: [" + str(i) + "] unlock", what_to_do)
                    else:
                        what_to_do[i, 0] = 0  # 锁定
                        print("ButtonA: [" + str(i) + "] lock", what_to_do)
                elif toc - tic > 1:  # 长按事件
                    keys_press_count[i, 1] += 1
                    if keys_press_count[i, 1] % 2 == 1:
                        what_to_do[i, 1] = 1  # 启动伺服
                        print("ButtonA: [" + str(i) + "] servo")
                    else:
                        what_to_do[i, 1] = 0  # 停止伺服
                        print("ButtonA: [" + str(i) + "] stop servo")

        # 处理按钮B的按下和释放事件
        for i in range(2):
            if dev_keys[i, 1] == -1:  # 按钮B按下
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1  # 开始录制
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        keys_press_count[0, 2] += 1
                else:
                    what_to_do[0, 2] = 0  # 停止录制
                    keys_press_count[0, 2] += 1

        last_keys_status = now_keys  # 更新上一次的按键状态

# 定义相机线程的全局变量
# 展开的图像
npy_list = np.array([np.zeros(480 * 640 * 3), np.zeros(480 * 640 * 3), np.zeros(480 * 640 * 3), np.zeros(480 * 640 * 3)])
# 图像长度
npy_len_list = np.array([0, 0, 0, 0])
# 图像列表
img_list = np.array([np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3))])
# 深度列表
depth_list = np.array([None, None, None, None])

# 相机线程函数
def run_thread_cam(rs_cam, which_cam):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]# 压缩图像，50是压缩质量，值越小，压缩比越大，图像质量越差
    while 1:
        # 读取图像和深度数据
        image_cam, depth_frame = rs_cam.read()  # 读取相机图像和深度数据
        
        # 处理图像
        image_cam = image_cam[:, :, ::-1]  # 将图像从BGR转换为RGB
        if which_cam == 3:  # 如果是bottom相机
            image_cam = cv2.flip(image_cam, 0)  # 上下翻转图像
            image_cam = cv2.flip(image_cam, 1)  # 左右翻转图像
        
        # img_list存储图像
        img_list[which_cam] = image_cam  # 存储图像
        _, image_ = cv2.imencode('.jpg', image_cam, encode_param)  # 压缩图像
        npy_list[which_cam][:len(image_)] = image_  # 存储压缩后的图像
        npy_len_list[which_cam] = len(image_)  # 存储图像长度

        # 存储深度数据
        depth_list[which_cam] = depth_frame  # 存储深度数据

# 主函数
def main(args):
    # 创建数据集文件路径
    save_dir = args.save_data_path + args.project_name + "/collect_data"
    mk_dir(save_dir)

    # 初始化相机
    # 从配置文件中加载相机的设备ID
    camera_dict = load_ini_data_camera()
    '''
    这里是通过configparser.ConfigParser()读取ini文件，获取相机的设备ID
    '''

    # 初始化顶部相机，设置图像翻转为True
        # RealSenseCamera类的构造函数通常需要两个参数：
        # - flip: 布尔值，指示是否翻转图像
        # - device_id: 相机的设备ID，用于标识连接的相机
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])  
    # 初始化左侧相机，不翻转图像
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    # 初始化右侧相机，设置图像翻转为True
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    # 初始化底部相机，设置图像翻转为True
    rs4 = RealSenseCamera(flip=True, device_id=camera_dict["bottom"])


    # 启动相机线
    # threading.Thread用于创建一个新的线程来运行指定的函数
    # target参数指定线程要运行的函数，args参数是给函数的参数
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_bottom = threading.Thread(target=run_thread_cam, args=(rs4, 3))

    # 启动线程，开始捕获图像
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_bottom.start()

    # 创建一个空白的画布，用于显示四个相机的图像
    # np.zeros创建一个全零的数组，形状为(480, 640 * 4, 3)，表示四个640x480的RGB图像
    show_canvas = np.zeros((480, 640 * 4, 3), dtype=np.uint8)

    # 暂停2秒，确保相机线程已初始化
    time.sleep(2)
    print("camera thread init success...")

    # 初始化机器人代理
    _, hands_dict = load_ini_data_hands()
    left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
    right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
    agent = BimanualAgent(left_agent, right_agent)

    # 初化机器人姿态
    print("Waiting to connect the robot...")
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    print("If the robot fails to initialize successfully after 5 seconds,please check that the robot network is connected correctly and make sure TCP/IP mode is turned!")
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    robot_pose_init(env)
    start_servo = False
    curr_light = "dark"
    print("robot init success....")

    # 初始化按钮状态
    last_status = np.array(([0, 0, 0], [0, 0, 0]))  # 初始化锁定状态
    thread_button = threading.Thread(target=button_monitor_realtime, args=(agent,))
    thread_button.start()
    print("button thread init success...")

    print("-------------------------Ok, let's start------------------------")
    idx = 0
    total_time = 0.04
    while 1:
        tic = time.time()
        action = agent.act({})  # 获取机器人动作
        print("action:")
        print(action)

        dev_what_to_do = what_to_do.copy() - last_status  # 计算按钮状态的变化
        last_status = what_to_do.copy()
        # 处理按钮A的短按事件（锁定和解锁）
        for i in range(2):
            if dev_what_to_do[i, 0] != 0:
                agent.set_torque(i, not what_to_do[i, 0])

        # 处理按钮A的长按事件（启动和停止伺服）
        if dev_what_to_do[0, 1] == 1 or dev_what_to_do[1, 1] == 1:
            print("dynamic approach")
            for i in range(2):
                if what_to_do[i, 1]:
                    agent.set_torque(i, True)
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])
            last_action = dynamic_approach(env, agent, flag_in)
            for i in range(2):
                if what_to_do[i, 0]:
                    if what_to_do[i, 1]:
                        agent.set_torque(i, False)
            start_servo = True
            obs = env.get_obs()
            if curr_light != "green":
                curr_light = set_light(env, "yellow", 1)

        if dev_what_to_do[0, 1] == -1 or dev_what_to_do[1, 1] == -1:
            flag_in = np.array([what_to_do[0, 1], what_to_do[1, 1]])

        if (what_to_do[0, 1] or what_to_do[1, 1]) and start_servo:
            action = agent.act({})
            err3, action = servo_action_check(action, last_action, flag_in)
            assert err3 != 0, set_light(env, "red", 1)

            # 安全保护逻辑
            protect_err = False
            delta = np.abs(action - last_action) / total_time
            t1 = time.time()
            pos = env.get_XYZrxryrz_state()
            print("pos:", pos)
            t2 = time.time()
            print("time:", t2 - t1)

            # 处理按钮B的录制事件
            if dev_what_to_do[0, 2] == 1:
                curr_light = set_light(env, "green", 1)
            elif dev_what_to_do[0, 2] == -1:
                curr_light = set_light(env, "yellow", 1)
            if what_to_do[0, 2] == 1:
                idx += 1
                # 创建文件夹
                left_dir = save_dir + f"/{dt_time[0]}/leftImg/"
                right_dir = save_dir + f"/{dt_time[0]}/rightImg/"
                top_dir = save_dir + f"/{dt_time[0]}/topImg/"
                bottom_dir = save_dir + f"/{dt_time[0]}/bottomImg/"
                depth_dir = save_dir + f"/{dt_time[0]}/depth/"
                mk_dir(right_dir)
                mk_dir(top_dir)
                mk_dir(bottom_dir)
                mk_dir(depth_dir)
                if mk_dir(left_dir):
                    idx = 0
                cv2.imwrite(top_dir + f"{idx}.jpg", img_list[0])
                cv2.imwrite(left_dir + f"{idx}.jpg", img_list[1])
                cv2.imwrite(right_dir + f"{idx}.jpg", img_list[2])
                cv2.imwrite(bottom_dir + f"{idx}.jpg", img_list[3])

                # 保存深度数据
                for i in range(4):
                    if depth_list[i] is not None:
                        np.save(depth_dir + f"{idx}_cam{i}.npy", depth_list[i])

                obs_dir = save_dir + f"/{dt_time[0]}/observation/"
                mk_dir(obs_dir)
                save_frame(obs_dir, idx, obs, action)

            obs = env.step(action, flag_in)
            obs["joint_positions"][6] = action[6]
            obs["joint_positions"][13] = action[13]
            last_action = action
        else:
            start_servo = False
            set_light(env, "green", 0)

        # 显示图像和深度信息
        if args.show_img:
            # 创建一个新的画布用于显示RGB和深度图像
            show_canvas = np.zeros((480 * 2, 640 * 4, 3), dtype=np.uint8)

            # 显示RGB图像
            show_canvas[:480, :640] = np.asarray(img_list[0], dtype="uint8")
            show_canvas[:480, 640:640 * 2] = np.asarray(img_list[1], dtype="uint8")
            show_canvas[:480, 640 * 2:640 * 3] = np.asarray(img_list[2], dtype="uint8")
            show_canvas[:480, 640 * 3:640 * 4] = np.asarray(img_list[3], dtype="uint8")

            # 将深度数据转换为彩色图像并显示
            for i in range(4):
                if depth_list[i] is not None:
                    # 将深度数据缩放到0-255范围
                    depth_scaled = cv2.convertScaleAbs(depth_list[i], alpha=0.03)
                    # 将灰度深度图转换为彩色
                    depth_colored = cv2.applyColorMap(depth_scaled, cv2.COLORMAP_JET)
                    # 将彩色深度图放在画布的下半部分
                    show_canvas[480:, i * 640:(i + 1) * 640] = depth_colored

            # 显示组合后的图像
            cv2.imshow("RGB and Depth", show_canvas)
            cv2.waitKey(1)

        toc = time.time()
        total_time = toc - tic
        print("total time: ", total_time)

if __name__ == "__main__":
    main(tyro.cli(Args))





