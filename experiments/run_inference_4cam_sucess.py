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
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.cameras.realsense_camera import RealSenseCamera
from scripts.manipulate_utils import load_ini_data_camera

# 修改导入语句
try:
    from module.model_module import Imitate_Model
except ImportError:
    print(f"无法导入 Imitate_Model。当前 Python 路径：")
    for path in sys.path:
        print(path)
    raise

# 定义命令行参数的数据类，包含机器人端口、主机名和是否显示图像的参数
@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True

# 定义全局变量，用于存储不同摄像头的图像数据、线程运行标志和线程锁
image_left, image_right, image_top, image_bottom, thread_run = None, None, None, None, None
lock = threading.Lock()# 线程锁: 用于保护共享资源，防止多个线程同时访问和修改共享资源

# 定义摄像头线程函数，根据摄像头索引读取图像数据
def run_thread_cam(rs_cam, which_cam):#传入摄像头对象和摄像头索引，例如：rs_cam = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    global image_left, image_right, image_top, image_bottom, thread_run
    while thread_run:
        image_cam, _ = rs_cam.read()  # 读取相机图像
        image_cam = image_cam[:, :, ::-1]  # BGR to RGB，与 run_control_4cam.py 一致
        
        if which_cam == 0:    # top camera
            image_top = image_cam
        elif which_cam == 1:  # left camera
            image_left = image_cam
        elif which_cam == 2:  # right camera
            image_right = image_cam
        elif which_cam == 3:  # bottom camera
            # 与 run_control_4cam.py 保持一致的底部相机图像翻转
            image_cam = cv2.flip(image_cam, 0)  # 上下翻转
            image_cam = cv2.flip(image_cam, 1)  # 左右翻转
            image_bottom = image_cam

# 主函数
def main(args):
    # 摄像头初始化
    global image_left, image_right, image_top, image_bottom, thread_run
    thread_run = True
    # 加载摄像头配置数据
    camera_dict = load_ini_data_camera()
    # 创建摄像头对象，设置是否翻转和设备 ID
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])      # 顶部相机
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])    # 左侧相机
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])    # 右侧相机
    rs4 = RealSenseCamera(flip=True, device_id=camera_dict["bottom"])   # 底部相机
    # 创建四个线程分别读取不同摄像头的图像数据【线程的意思是同时执行多个任务，而不是一个任务执行完再执行下一个任务】
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_bottom = threading.Thread(target=run_thread_cam, args=(rs4, 3))
    # 启动线程
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_bottom.start()
    show_canvas = np.zeros((480, 640 * 4, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")
    # 机械臂初始化
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    print("robot init success...")
    # 移动机械臂到安全位置
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))
    time.sleep(1)
    # 移动机械臂到初始拍照位置
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))


    #10.24测试 非常顺利
    #ckpt_dir = './1007_plasticbag_L_300_task' 
    #ckpt_dir = './ckpt/yellow_plastic_lrb_task1013xunlian' 
    #11.10测试 不顺利
    #chunk10  
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk10' 
    #11.12测试 不顺利
    #chunk30  
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk30'   model_name = 'policy_step_10000_seed_0.ckpt' X 找错塑料袋
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk30'   model_name = 'policy_step_20000-30000_seed_0.ckpt' X 触底
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk30'   model_name = 'policy_step_40000-60000_seed_0.ckpt' X 从底部不抬起
    #chunk25
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk25'   model_name = 'policy_step_10000_seed_0.ckpt' X 触地
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk25'   model_name = 'policy_step_20000-30000_seed_0.ckpt' X 找错塑料袋
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk25'   model_name = 'policy_step_40000_seed_0.ckpt' X 从底部不抬起
    #50000/60000/100000一样的问题
    #11.14测试 不顺利
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk30_modfprms' model_name = 'policy_step_40000_seed_0.ckpt' X触底
    #ckpt_dir = './ckpt/1104_collect100_yellow_plasticbag_M_chunk30_modfprms' model_name = 'policy_step_50000_seed_0.ckpt' X触底
    #11.18测试 顺利 测试抓小型
    #ckpt_dir = './ckpt/1118_yellow_plasticbag_M_testfeasibility' model_name = 'policy_last.ckpt'
    #11.25测试 顺利顺利！！！
    #ckpt_dir = './ckpt/1119_yellow_plasticbag_M_onlygrabbag' model_name = 'policy_last.ckpt'

    #加载模型
    model_name = 'policy_step_40000_seed_0.ckpt'
    print('测试点A：准备加载模型')
    #1119_yellow_plasticbag_M_onlygrabbag_chunk60    好用
    #1217_yellow_plasticbag_M_3toys_chunk100
    model = Imitate_Model(ckpt_dir='./ckpt/1119_yellow_plasticbag_M_onlygrabbag_chunk60', ckpt_name=model_name)
    print("模型路径：",model.ckpt_dir)
    print(model.camera_names)
    print(model.policy_config)
    print('测试点B：正在加载模型')
    model.loadModel()
    print("model init success...")

    # 初始化任务参数
    episode_len = 4000
    t = 0
    last_time = 0
    # 定义观测数据的字典，包含关节位置和不同视角的图像
    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': [], 'bottom': []}}
    obs = env.get_obs()
    # 设置夹爪的初始位置
    obs["joint_positions"][6] = 1.0
    obs["joint_positions"][13] = 1.0
    observation['qpos'] = obs["joint_positions"]
    last_action = observation['qpos'].copy()

    first = True

    print("The robot begins to perform tasks autonomously...")
    while t < episode_len:
        time0 = time.time()
        # 构建观测数据，确保与 HDF5 文件中的格式完全一致
        observation = {
            'qpos': env.get_obs()["joint_positions"],#获得当前关节位置存入qpos中
            'images': {
                'top': image_top,           # image_top 是顶部相机的图像数据，存入top
                'left_wrist': image_left,   # image_left 是左侧相机的图像数据，存入left_wrist
                'right_wrist': image_right, # image_right 是右侧相机的图像数据，存入right_wrist
                'bottom': image_bottom      # image_bottom 是底部相机的图像数据，存入bottom
            }
        }
        # 显示图像
        if args.show_img:
            # 保持与 run_control_4cam.py 相同的显示顺序
            show_canvas = np.zeros((480, 640 * 4, 3), dtype=np.uint8)
            show_canvas[:, :640] = observation['images']['top']
            show_canvas[:, 640:640*2] = observation['images']['left_wrist']
            show_canvas[:, 640*2:640*3] = observation['images']['right_wrist']
            show_canvas[:, 640*3:640*4] = observation['images']['bottom']
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        time1 = time.time()
        print("read images time(ms)：", (time1 - time0) * 1000)

        # 模型推理，确保输入格式与训练时一致
        action = model.predict(observation, t)#输入observation和t，输出action
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



        # ×××××××××××××××××××××××××××××Security protection××××××××××××××××××××××××××××××××××××××××××××××��××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××
        # [Note]: Modify the protection parameters in this section carefully !
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
        # 检查左右机械臂关节角度是否超出安全范围
        if not ((action[2] > -2.7 and action[2] < 0 and action[3] > -0.6) and \
                (action[9] < 2.7 and action[9] > 0 and action[10] < 0.6)):
            print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
            print(action)
            protect_err = True

        # 检查机械臂位置是否超出安全范围
        t1 = time.time()
        pos = env.get_XYZrxryrz_state()
        #修改pos[8]限位42->28
        if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 20) and \
                (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 20)):
        # if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 42) and \
        #         (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 42)):
            print("[Warn]:The robot arm XYZ is out of the safe position! ")
            print(pos)
            protect_err = True
        t2 = time.time()
        print("get pos time(ms):", (t2 - t1) * 1000)

        if protect_err:
            env.set_do_status([3, 0])
            env.set_do_status([2, 0])
            env.set_do_status([1, 1])
            time.sleep(1)
            exit()
        # ×××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××××xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx




        # 如果first为True，则进行关节位置的平滑过渡
        if first:
            max_delta = (np.abs(last_action - action)).max()#计算关节位置的最大变化量
            steps = min(int(max_delta / 0.001), 150)#计算关节位置变化的步数
            for jnt in np.linspace(last_action, action, steps):#在关节位置之间进行平滑过渡
                env.step(jnt, np.array([1, 1]))
            first = False

        last_action = action.copy()#更新关节位置

       
        # 控制机器人运动
        time3 = time.time()
        #本来是np.array([1, 1])，现在改成np.array([0.6, 0.6])，降低运动速度
        obs = env.step(action, np.array([0.6, 0.6]))#输入模型预测得到的action，控制机器人运动
        time4 = time.time()


        ################修改成点动###############
        print("Press any key to continue...")
        cv2.waitKey(0)


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