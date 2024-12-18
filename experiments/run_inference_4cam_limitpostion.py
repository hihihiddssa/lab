import sys
import os

# 获取当前文件的上级目录的上级目录，并添加 /ModelTrain/ 构建基础目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/ModelTrain/"
# 将构建好的基础目录路径添加到 Python 模块搜索路径中
sys.path.append(BASE_DIR)

# 模型配置
MODEL_CONFIG = {
    'ckpt_dir': './ckpt/1104_collect100_yellow_plasticbag_M_chunk30',  # 模型检查点目录
    'ckpt_name': 'policy_step_80000_seed_0.ckpt'  # 模型检查点文件名
}
MIN_Z_HEIGHT = 35.0
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

# 定义命令行参数的数据类
@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True

# 定义全局变量，用于存储不同摄像头的图像数据和线程运行标志
image_left, image_right, image_top, image_bottom, thread_run = None, None, None, None, None
lock = threading.Lock()

# 相机线程函数
def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, image_bottom, thread_run
    while thread_run:
        image_cam, _ = rs_cam.read()  # 读取相机图像
        image_cam = cv2.cvtColor(image_cam, cv2.COLOR_BGR2RGB)  # BGR 转 RGB，与 run_control_4cam.py 保持一致
        
        if which_cam == 0:    # top camera
            image_top = image_cam
        elif which_cam == 1:  # left camera
            image_left = image_cam
        elif which_cam == 2:  # right camera
            image_right = image_cam
        elif which_cam == 3:  # bottom camera
            # 只对底部相机进行翻转处理
            image_cam = cv2.flip(image_cam, 0)  # 上下翻转
            image_cam = cv2.flip(image_cam, 1)  # 左右翻转
            image_bottom = image_cam

def main(args):
    # 摄像头初始化
    global image_left, image_right, image_top, image_bottom, thread_run
    thread_run = True
    
    # 加载摄像头配置数据
    camera_dict = load_ini_data_camera()
    
    # 初始化相机，与 run_control_4cam.py 保持一致
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])      # 顶部相机
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])    # 左侧相机
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])    # 右侧相机
    rs4 = RealSenseCamera(flip=True, device_id=camera_dict["bottom"])   # 底部相机

    # 创建四个线程分别读取不同摄像头的图像数据
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

    # 机器人初始化
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client)
    env.set_do_status([1, 0])
    env.set_do_status([2, 0])
    env.set_do_status([3, 0])
    print("robot init success...")

    # 移动机器人到安全位置
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
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
    print('测试点A：准备加载模型')
    model = Imitate_Model(
        ckpt_dir=MODEL_CONFIG['ckpt_dir'], 
        ckpt_name=MODEL_CONFIG['ckpt_name']
    )
    print(model.camera_names)
    print(model.policy_config)

    print('测试点B：正在加载模型')
    model.loadModel()
    print("model init success...")

    # 初始化任务参数
    episode_len = 4000
    t = 0
    last_time = 0
    observation = {'qpos': [], 'images': {'left_wrist': [], 'right_wrist': [], 'top': [], 'bottom': []}}
    obs = env.get_obs()
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
            'qpos': env.get_obs()["joint_positions"],
            'images': {
                'top': image_top,           # 对应 img_list[0]
                'left_wrist': image_left,   # 对应 img_list[1]
                'right_wrist': image_right, # 对应 img_list[2]
                'bottom': image_bottom      # 对应 img_list[3]
            }
        }

        if args.show_img:
            # 保持与 run_control_4cam.py 相同的显示顺序
            show_canvas[:, :640] = observation['images']['top']
            show_canvas[:, 640:640*2] = observation['images']['left_wrist']
            show_canvas[:, 640*2:640*3] = observation['images']['right_wrist']
            show_canvas[:, 640*3:640*4] = observation['images']['bottom']
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        time1 = time.time()
        print("read images time(ms)：", (time1 - time0) * 1000)

        # 模型推理，确保输入格式与训练时一致
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

        # 在执行动作前，先计算该动作导致的位置，并进行安全检查
        MIN_Z_HEIGHT = 35.0  # 定义统一的最小安全高度
        def check_and_adjust_position(action):
            """检查并调整机械臂位置，确保不会低于安全高度"""
            current_pos = env.get_XYZrxryrz_state()
            current_joints = env.get_obs()["joint_positions"]
            adjusted = False
            
            # 检查左臂 Z 轴
            if current_pos[2] <= MIN_Z_HEIGHT:
                print(f"[Warning] Left arm Z position ({current_pos[2]:.2f}) below safety height")
                action[:6] = current_joints[:6]  # 保持当前左臂关节角度
                adjusted = True
                
            # 检查右臂 Z 轴
            if current_pos[8] <= MIN_Z_HEIGHT:
                print(f"[Warning] Right arm Z position ({current_pos[8]:.2f}) below safety height")
                action[7:13] = current_joints[7:13]  # 保持当前右臂关节角度
                adjusted = True
            
            if adjusted:
                print("[Info] Action adjusted to maintain safe height")
            
            return action

        # 获取模型预测的动作后，进行位置调整
        adjusted_action = check_and_adjust_position(action)
        if adjusted_action is None:
            print("[Warn]: Position adjustment failed, skipping action")
            continue
        
        action = adjusted_action

        # 安全保护检查
        protect_err = False
        delta = action - last_action
        print("Joint increment：", delta)
        
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

        if not ((action[2] > -2.6 and action[2] < 0 and action[3] > -0.6) and \
                (action[9] < 2.6 and action[9] > 0 and action[10] < 0.6)):
            print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
            print(action)
            protect_err = True

        t1 = time.time()
        pos = env.get_XYZrxryrz_state()
        
        # 如果Z轴低于安全高度，则将其限制在安全高度
        
        if pos[2] < MIN_Z_HEIGHT:
            print(f"[Info] Left arm Z position adjusted from {pos[2]:.2f} to {MIN_Z_HEIGHT}")
            pos[2] = MIN_Z_HEIGHT
        if pos[8] < MIN_Z_HEIGHT:
            print(f"[Info] Right arm Z position adjusted from {pos[8]:.2f} to {MIN_Z_HEIGHT}")
            pos[8] = MIN_Z_HEIGHT
            
        # 检查XYZ是否在安全范围内
        if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 28) and \
                (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 28)):
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

        # 打印当前位置信息
        print("当前action是:", action)
        current_pos = env.get_XYZrxryrz_state()
        print("当前pos是:", current_pos)

        ################点动功能###############
        print("Press any key to continue...")
        key = cv2.waitKey(0)
        # 按 'q' 或 'ESC' 键退出程序
        if key == ord('q') or key == 27:  # 27 是 ESC 键的 ASCII 码
            print("Program terminated by user")
            thread_run = False
            cv2.destroyAllWindows()
            break
        ################点动功能###############

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

        # 更新机器人的观测数据
        obs["joint_positions"][6] = action[6]
        obs["joint_positions"][13] = action[13]
        observation['qpos'] = obs["joint_positions"]

        print("Read joint value time(ms)：", (time4 - time3) * 1000)
        t += 1
        print("The total time(ms):", (time4 - time0) * 1000)

    thread_run = False
    print("Task accomplished")

if __name__ == "__main__":
    main(tyro.cli(Args))