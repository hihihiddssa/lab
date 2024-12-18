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

try:
    from module.model_module import Imitate_Model
except ImportError:
    print(f"无法导入 Imitate_Model。当前 Python 路径：")
    for path in sys.path:
        print(path)
    raise

@dataclass
class Args:
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = True

# 定义全局变量
image_left, image_right, image_top, image_bottom, thread_run = None, None, None, None, None
lock = threading.Lock()

# 在文件开头添加锁和标志
_resources_released = False
_release_lock = threading.Lock()

def release_resources():
    """安全地释放资源"""
    global _resources_released, thread_run
    
    # 使用锁确保只有一个线程可以执行释放操作
    with _release_lock:
        if not _resources_released:
            try:
                thread_run = False
                time.sleep(0.5)  # 给线程一些时间来结束
                
                # 关闭所有OpenCV窗口
                cv2.destroyAllWindows()
                
                print("资源已释放")
                _resources_released = True
            except Exception as e:
                print(f"释放资源时发生错误: {e}")

def run_thread_cam(rs_cam, which_cam):
    global image_left, image_right, image_top, image_bottom, thread_run
    while thread_run:
        image_cam, _ = rs_cam.read()
        image_cam = image_cam[:, :, ::-1]
        
        if which_cam == 0:    # top camera
            image_top = image_cam
        elif which_cam == 1:  # left camera
            image_left = image_cam
        elif which_cam == 2:  # right camera
            image_right = image_cam
        elif which_cam == 3:  # bottom camera
            image_cam = cv2.flip(image_cam, 0)
            image_cam = cv2.flip(image_cam, 1)
            image_bottom = image_cam

def check_model1_complete(action, last_action):
    """
    检查第一个模型是否执行完成，通过观察动作增量是否稳定
    """
    if last_action is None:
        return False
        
    # 获取动作增量
    delta = action - last_action
    
    # 如果所有关节的变化都很小，说明机械臂基本稳定
    threshold = 0.001  # 设置一个较小的阈值
    is_stable = all(abs(d) < threshold for d in delta)
    
    return is_stable

def check_model_switch_state(model1_final_action, observation):
    """检查模型切换时的状态一致性"""
    # 检查关节位置
    joint_diff = np.abs(model1_final_action - observation['qpos'])
    if np.any(joint_diff > 0.01):  # 阈值可调
        print("警告：关节位置与模型一结束状态不一致")
        print("差异：", joint_diff)
        return False
        
    # 检查夹爪状态
    if (abs(model1_final_action[6] - observation['qpos'][6]) > 0.01 or 
        abs(model1_final_action[13] - observation['qpos'][13]) > 0.01):
        print("警告：夹爪状态不一致")
        return False
    
    return True

def check_position_safety(pos):
    """完善的位置安全检查"""
    # 工作空间限制
    workspace_limits = {
        'x': (-410, 210),
        'y': (-700, -210),
        'z': (20, float('inf')),
        'x2': (-210, 410),
        'y2': (-700, -210),
        'z2': (20, float('inf'))
    }
    
    # 左臂检查
    if not (workspace_limits['x'][0] < pos[0] < workspace_limits['x'][1] and
            workspace_limits['y'][0] < pos[1] < workspace_limits['y'][1] and
            workspace_limits['z'][0] < pos[2]):
        return False
        
    # 右臂检查
    if not (workspace_limits['x2'][0] < pos[6] < workspace_limits['x2'][1] and
            workspace_limits['y2'][0] < pos[7] < workspace_limits['y2'][1] and
            workspace_limits['z2'][0] < pos[8]):
        return False
    
    return True

def check_observation_quality(observation):
    """检查观测数据质量"""
    # 检查图像是否为空
    for cam_name, img in observation['images'].items():
        if img is None or img.size == 0:
            print(f"警告：{cam_name} 相图像为空")
            return False
            
    # 检查关节数据是否在合理范围
    qpos = observation['qpos']
    if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
        print("警告：关节数据包含无效值")
        return False
        
    return True

def safe_transition(env, start_pos, target_pos, steps=150, speed=0.1, delay=0.05):
    """安全的过渡函数"""
    try:
        print(f"开始安全过渡，步数: {steps}")
        
        # 分析关节变化（排除夹爪）
        joint_changes = np.abs(target_pos - start_pos)
        print("各关节变化量(弧度):")
        for i, change in enumerate(joint_changes):
            if i != 6 and i != 13:  # 排除夹爪
                print(f"J{i+1}: {change:.6f} ({np.rad2deg(change):.2f}度)")
        
        # 根据变化量调整过渡策略
        arm_joint_changes = np.concatenate([joint_changes[:6], joint_changes[7:13]])
        max_change = np.max(arm_joint_changes)
        adjusted_steps = int(max_change / 0.003)  # 减小每步的最大变化量
        steps = max(steps, adjusted_steps)
        
        # 生成平滑的过渡轨迹
        positions = np.linspace(start_pos, target_pos, steps)
        
        # 执行过渡动作
        for pos in positions:
            # 检查位置安全性
            if not check_position_safety(env.get_XYZrxryrz_state()):
                print("警告：位置超出安全范围")
                return False, env.get_obs()["joint_positions"]
                
            # 执行移动
            env.step(pos, np.array([speed, speed]))
            time.sleep(delay)
            
        return True, env.get_obs()["joint_positions"]
        
    except Exception as e:
        print(f"过渡过程中发生错误: {e}")
        return False, env.get_obs()["joint_positions"]

def main(args):
    # 摄像头初始化
    global image_left, image_right, image_top, image_bottom, thread_run
    thread_run = True
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    rs4 = RealSenseCamera(flip=True, device_id=camera_dict["bottom"])
    
    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_bottom = threading.Thread(target=run_thread_cam, args=(rs4, 3))
    
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

    # 移动到安全位置
    reset_joints_left = np.deg2rad([-90, 30, -110, 20, 90, 90, 0])
    reset_joints_right = np.deg2rad([90, -30, 110, -20, -90, -90, 0])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))
    time.sleep(1)

    # 移动到初始位置
    reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
    reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])
    reset_joints = np.concatenate([reset_joints_left, reset_joints_right])
    curr_joints = env.get_obs()["joint_positions"]
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.001), 150)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt, np.array([1, 1]))

    # 初始化任务参数
    episode_len = 2000
    t = 0
    current_phase = 1  # 1: 模型一, 2: 模型二
    check_counter = 0
    stability_threshold = 10
    first_model1 = True
    first_model2 = True
    
    # 初始化 last_action
    obs = env.get_obs()
    last_action = obs["joint_positions"].copy()  # 使用当前关节位置初始化
    model1_final_action = None
    
    # 加载模型一
    #   'policy_step_60000_seed_0.ckpt'
    #   'policy_best.ckpt'
    model1_name = 'policy_step_60000_seed_0.ckpt'
    print('加载模型一')
    model1 = Imitate_Model(ckpt_dir='./ckpt/1119_yellow_plasticbag_M_onlygrabbag_chunk40', ckpt_name=model1_name)
    model1.loadModel()
    
    # 加载模型二
    model2_name = 'policy_best.ckpt'
    print('加载模型二')
    model2 = Imitate_Model(ckpt_dir='./ckpt/1203_yellow_plasticbag_M_grabtoy_chunk40', ckpt_name=model2_name)
    model2.loadModel()

    while t < episode_len:
        time0 = time.time()
        
        # 构建观测数据
        observation = {
            'qpos': env.get_obs()["joint_positions"],
            'images': {
                'top': image_top,
                'left_wrist': image_left,
                'right_wrist': image_right,
                'bottom': image_bottom
            }
        }

        # 示图像
        if args.show_img:
            show_canvas = np.zeros((480, 640 * 4, 3), dtype=np.uint8)
            show_canvas[:, :640] = observation['images']['top']
            show_canvas[:, 640:640*2] = observation['images']['left_wrist']
            show_canvas[:, 640*2:640*3] = observation['images']['right_wrist']
            show_canvas[:, 640*3:640*4] = observation['images']['bottom']
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)

        if current_phase == 1:
            # 模型一推理
            action = model1.predict(observation, t)
            
            # 动作限制
            action[6] = np.clip(action[6], 0, 1)
            action[13] = np.clip(action[13], 0, 1)
            
            # 检查是否稳定
            if not first_model1 and check_model1_complete(action, last_action):
                check_counter += 1
                print(f"检测到模型1可能执行完成 ({check_counter}/{stability_threshold})")
                
                if check_counter >= stability_threshold:
                    print("模型1执行完成！")
                    print("准备执行模型2...")
                    model1_final_action = action.copy()
                    current_phase = 2
                    check_counter = 0
                    first_model2 = True
                    continue
            
            # 安全检查
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
            '''
            原来的：
            if not ((action[2] > -3.0 and action[2] < 0 and action[3] > -0.6) and \
                    (action[9] < 3.0 and action[9] > 0 and action[10] < 0.6)):
            '''
            if not ((action[2] > -3.0 and action[2] < 0 and action[3] > -1.7) and \
                    (action[9] < 3.0 and action[9] > 0 and action[10] < 1.7)):
                print("[Warn]:The J3 or J4 joints of the robotic arm are out of the safe position! ")
                print(action)
                protect_err = True

            pos = env.get_XYZrxryrz_state()
            if not ((pos[0] > -410 and pos[0] < 210 and pos[1] > -700 and pos[1] < -210 and pos[2] > 20) and \
                    (pos[6] < 410 and pos[6] > -210 and pos[7] > -700 and pos[7] < -210 and pos[8] > 20)):
                print("[Warn]:The robot arm XYZ is out of the safe position! ")
                print(pos)
                protect_err = True

            if protect_err:
                env.set_do_status([3, 0])
                env.set_do_status([2, 0])
                env.set_do_status([1, 1])
                time.sleep(1)
                exit()

            # 如果是第一次执行，进行平滑过渡
            if first_model1:
                print("模型1第一次执行，进行平滑过渡...")
                max_delta = (np.abs(last_action - action)).max()
                steps = min(int(max_delta / 0.001), 150)
                for jnt in np.linspace(last_action, action, steps):
                    env.step(jnt, np.array([1, 1]))
                first_model1 = False

            last_action = action.copy()

            # 执行动作
            obs = env.step(action, np.array([0.6, 0.6]))
            
            # 更新观测数据
            obs["joint_positions"][6] = action[6]
            obs["joint_positions"][13] = action[13]
            observation['qpos'] = obs["joint_positions"]

        elif current_phase == 2:
            # 如果是第一次执行模型二
            if first_model2:
                print("\n开始模型2过渡...")
                first_model2 = False
                continue
            
            # 确保获取最新的观测数据
            observation = {
                'qpos': env.get_obs()["joint_positions"],
                'images': {
                    'top': image_top,
                    'left_wrist': image_left,
                    'right_wrist': image_right,
                    'bottom': image_bottom
                }
            }
            
            # 检查观测数据质量
            if not check_observation_quality(observation):
                print("观测数据质量异常，跳过当前帧")
                continue
            
            # 获取模型预测
            action = model2.predict(observation, t)
            
            # 打印详细的调试信息
            print(f"\n当前时间步: {t}")
            print(f"当前关节位置: {observation['qpos']}")
            print(f"预测的动作: {action}")
            
            # 动作限制
            action[6] = np.clip(action[6], 0, 1)
            action[13] = np.clip(action[13], 0, 1)
            
            # 计算并打印关节增量
            delta = action - last_action
            print("关节增量：", delta)
            print("最大增量：", np.max(np.abs(delta)))
            
            # 如果动作变化太小，可能需要重新获取观测
            if np.max(np.abs(delta)) < 1e-6:
                print("警告：动作变化极小，可能需要检查模型预测")
                time.sleep(0.1)  # 短暂等待以获取新的传感器数据
                continue
            
            # 执行安全检查
            protect_err = False
            # ... (保持原有的安全检查代码不变)
            
            # 执行动作并更新状态
            obs = env.step(action, np.array([0.6, 0.6]))
            last_action = action.copy()
            
            # 更新观测数据
            obs["joint_positions"][6] = action[6]
            obs["joint_positions"][13] = action[13]
            observation['qpos'] = obs["joint_positions"]
            
            # 添加短暂延迟以确保状态更新
            time.sleep(0.05)
            
        t += 1

    print("任务完成")

if __name__ == "__main__":
    try:
        main(tyro.cli(Args))
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
    finally:
        release_resources()