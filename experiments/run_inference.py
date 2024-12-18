import sys
import os

# 获取当前文件的上级目录的上级目录，并添加 /ModelTrain/ 构建基础目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ModelTrain/"
# 将构建好的基础目录路径添加到 Python 模块搜索路径中
sys.path.append(BASE_DIR)

import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
import queue
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.cameras.realsense_camera import RealSenseCamera
from scripts.manipulate_utils import load_ini_data_camera
from ModelTrain.module.model_module import Imitate_Model



#action是关节角度


# 定义命令行参数的数据类，包含机器人端口、主机名和是否显示图像的参数
@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True


# 定义全局变量，用于存储不同摄像头的图像数据、线程运行标志和线程锁
image_left, image_right, image_top, thread_run = None, None, None, None
lock = threading.Lock()


# 定义摄像头线程函数，根据摄像头索引读取图像数据
def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, thread_run
    if which_cam == 0:
        while thread_run:
            # 读取图像并转换颜色通道顺序
            image_left, _ = rs_cam.read()
            image_left = image_left[:, :, ::-1]
    elif which_cam == 1:
        while thread_run:
            image_right, _ = rs_cam.read()
            image_right = image_right[:, :, ::-1]
    elif which_cam == 2:
        while thread_run:
            image_top, _ = rs_cam.read()
            image_top = image_top[:, :, ::-1]
    else:
        print("Camera index error! ")


# 主函数
def main(args):
    # 摄像头初始化  # camera init
    global image_left, image_right, image_top, thread_run
    thread_run = True
    # 加载摄像头配置数据
    camera_dict = load_ini_data_camera()
    # 创建左侧摄像头对象，设置是否翻转和设备 ID
    rs1 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    # 创建右侧摄像头对象，设置是否翻转和设备 ID
    rs2 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    # 创建顶部摄像头对象，设置是否翻转和设备 ID
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    # 创建三个线程分别读取不同摄像头的图像数据
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    # 启动线程
    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_top.start()
    show_canvas = np.zeros((480, 640 * 3, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

    # 机器人初始化  # robot init
    # 创建与机器人通信的客户端对象，指定端口和主机名
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    # 创建机器人环境对象
    env = RobotEnv(robot_client)
    # 设置机器人数字输出状态
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    print("robot init success...")

    # 移动机器人到安全位置
    # 定义左侧机器人关节角度（单位为弧度）
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    # 定义右侧机器人关节角度（单位为弧度）
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    # 合并左右侧关节角度
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    # 获取当前机器人关节位置
    curr_joints = env.get_obs()["joint_positions"]
    # 计算与目标位置的最大差值
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    # 根据最大差值计算移动步数
    steps = min(int(max_delta / 0.001), 150)
    # 逐步移动机器人到安全位置
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))
    time.sleep(1)

    # 移动机器人到初始拍照位置
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))

    # 模型初始化
    model_name = 'policy_step_82000_seed_0.ckpt'
    model = Imitate_Model(ckpt_dir='./ckpt/1104_collect100_yellow_plasticbag_M_3cam_useversion4cam', ckpt_name=model_name)
    model.loadModel()
    print("model init success...")

    # 初始化任务参数
    episode_len = 1000
    t = 0
    last_time = 0
    # 定义观测数据的字典，包含关节位置和不同视角的图像
    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': []}}
    obs = env.get_obs()
    # 设置夹爪的初始位置
    obs["joint_positions"][6] = 1.0
    obs["joint_positions"][13] = 1.0
    observation['qpos'] = obs["joint_positions"]
    last_action = observation['qpos'].copy()

    first = True

    print("The robot begins to perform tasks autonomously...")
    while t < episode_len:
        # 获取当前图像数据
        time0 = time.time()
        observation['images']['left_wrist'] = image_left
        observation['images']['right_wrist'] = image_right
        observation['images']['top'] = image_top
        if args.show_img:
            # 拼接图像并显示
            imgs = np.hstack((observation['images']['left_wrist'], observation['images']['right_wrist'],
                              observation['images']['top']))
            cv2.imshow("imgs", imgs)
            cv2.waitKey(1)
        time1 = time.time()
        print("read images time(ms)：", (time1 - time0) * 1000)

        # 模型推理，得到控制机器人运动的关节角度值
        action = model.predict(observation, t)
        if action[6] > 1:
            action[6] = 1
        elif action[6] < 0:
            action[6] = 0
        if action[13] > 1:
            action[13] = 1
        elif action[13] < 0:
            action[13] = 0
        time2 = time.time()
        print("Model inference time(ms)：", (time2 - time1) * 1000)



        # ×××××××××××××××××××××××××××××Security protection×××××××××××××××××××××××××××××××××××××××××××
        # [Note]: Modify the protection parameters in this section carefully !
        # 安全保护
        protect_err = False
        delta = action - last_action
        print("Joint increment：", delta)
        # 检查关节增量是否过大
        if max(delta[0:6]) > 0.17 or max(delta[7:13]) > 0.17:
            print("Note!If the joint increment is larger than 10 degrees!!!")
            print("Do you want to continue running?Press the 'Y' key to continue, otherwise press the other button to stop the program!")
            temp_img = np.zeros(shape=(640, 480))
            cv2.imshow("waitKey", temp_img)
            key = cv2.waitKey(0)
            if key == ord('y') or key == ord('Y'):
                cv2.destroyWindow("waitKey")
                max_delta = (np.abs(last_action - action)).max()
                steps = min(int(max_delta / 0.001), 100)
                for jnt in np.linspace(last_action, action, steps):
                    env.step(jnt, np.array([1, 1]))
            else:
                protect_err = True
                cv2.destroyAllWindows()
        # Left arm joint angle limitations:  -150<J3<0    J4>-35  (Note: This angle needs to be converted to radians)
        # right arm joint angle limitations:  150>J3>0    J4<35   (Note: This angle needs to be converted to radians)

        # 检查左右机械臂关节角度是否超出安全范围
        if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and \
                (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):
            print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
            print(action)
            protect_err = True

        # left arm (jaw tip position) limit:  210>x>-410  -700<Y<-210  z>47;
        # right arm (jaw tip position) limit:  410>x>-210  -700<Y<-210  z>47;



        # 检查机械臂位置是否超出安全范围
        t1 = time.time()
        pos = env.get_XYZrxryrz_state()
        if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 42) and \
                (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 42)):
            print("[Warn]:The robot arm XYZ is out of the safe position! ")
            print(pos)
            protect_err = True
        t2 = time.time()
        print("get pos time(ms):", (t2 - t1) * 1000)

        # ###后加入的代码（起点）
        # # 限制位置代码
        # # 定义左右机械臂 Z 坐标的限位值
        # z_limit_left = 100
        # z_limit_right = 100
        # # 获取机械臂位置状态
        # pos = env.get_XYZrxryrz_state()
        # # 对左机械臂 Z 坐标进行限制
        # if pos[2] > z_limit_left:
        #     action[2] = z_limit_left
        # # 对右机械臂 Z 坐标进行限制
        # if pos[8] > z_limit_right:
        #     action[9] = z_limit_right
        # ###后加入的代码（终点）

        if protect_err:
            env.set_do_status([3, 0])
            env.set_do_status([2, 0])
            env.set_do_status([1, 1])
            time.sleep(1)
            exit()
        # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××


        if first:
            max_delta = (np.abs(last_action - action)).max()
            steps = min(int(max_delta / 0.001), 100)
            for jnt in np.linspace(last_action, action, steps):
                env.step(jnt, np.array([1, 1]))
            first = False

        last_action = action.copy()

        # 控制机器人运动
        time3 = time.time()
        obs = env.step(action, np.array([1, 1]))
        time4 = time.time()

        # 更新机器人的观测数据  # Obtain the current joint value of the robots (including the gripper)
        obs["joint_positions"][6] = action[6]
        obs["joint_positions"][13] = action[13]
        observation['qpos'] = obs["joint_positions"]

        print("Read joint value time(ms)：", (time4 - time3) * 1000)
        t += 1
        print("The total time(ms):", (time4 - time0) * 1000)

    thread_run = False
    print("Task accomplished")

    # 返回起始位置（代码省略）
    #...


if __name__ == "__main__":
    main(tyro.cli(Args))

