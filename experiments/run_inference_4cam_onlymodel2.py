import sys
import os
import keyboard

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
        print("警告：关节位置与模型一���不一致")
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

def setup_initial_position(env):
    print("\n=== 关节位置设置模式 ===")
    print("使用以下按键控制关节:")
    print("←/→: 切换左/右机械臂")
    print("0-6: 控制当前机械臂的关节0-6")
    print("↑/↓: 按住可持续增加/减少选中关节的角度")
    print("[/]: 调整步长")
    print("Space: 完成设置")
    print("Esc: 退出程序")
    
    current_pos = env.get_obs()["joint_positions"].copy()
    print("\n当前位置:", current_pos)
    
    # 创建一个窗口用于接收键盘输入
    cv2.namedWindow("Joint Control")
    
    # 左右机械臂的关节映射
    joint_keys = {
        ord('0'): 0, ord('1'): 1, ord('2'): 2, ord('3'): 3, 
        ord('4'): 4, ord('5'): 5, ord('6'): 6
    }
    
    selected_joint = 0
    step_size = 0.005  # 基础步长
    is_right_arm = False  # False为左臂，True为右臂
    
    while True:
        # 示前��制状态
        display_img = np.zeros((200, 600, 3), dtype=np.uint8)
        arm_text = "Right Arm" if is_right_arm else "Left Arm"
        cv2.putText(display_img, f"Control: {arm_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        actual_joint = selected_joint + (7 if is_right_arm else 0)
        cv2.putText(display_img, f"Joint: {actual_joint}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_img, f"Position: {current_pos[actual_joint]:.4f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(display_img, f"Step size: {step_size:.4f}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Joint Control", display_img)
        
        key = cv2.waitKey(10) & 0xFF
        
        # Esc键退出
        if key == 27:  # Esc
            print("\n程序已退出")
            cv2.destroyWindow("Joint Control")
            return None
            
        # 空格键完成设置    
        if key == 32:  # Space
            print("\n设置完成！最终位置:", current_pos)
            cv2.destroyWindow("Joint Control")
            return current_pos
            
        # 切换左右机械臂
        if key == 81:  # Left arrow
            is_right_arm = False
            print("\n切换到左臂控制")
        elif key == 83:  # Right arrow
            is_right_arm = True
            print("\n切换到右臂控制")
            
        # 选择关节
        if key in joint_keys:
            selected_joint = joint_keys[key]
            actual_joint = selected_joint + (7 if is_right_arm else 0)
            print(f"\n当前选中关节 {actual_joint}")
            
        # 调整关节角度（持续按住）
        if key == 82:  # Up arrow
            actual_joint = selected_joint + (7 if is_right_arm else 0)
            # 对夹爪关节进行限制
            if actual_joint in [6, 13]:
                current_pos[actual_joint] = min(1.0, current_pos[actual_joint] + step_size)
            else:
                current_pos[actual_joint] += step_size
            print(f"关节 {actual_joint} 增加到: {current_pos[actual_joint]:.4f}")
            env.step(current_pos, np.array([0.2, 0.2]))
            time.sleep(0.02)
            
        if key == 84:  # Down arrow
            actual_joint = selected_joint + (7 if is_right_arm else 0)
            # 对夹爪关节进行限制
            if actual_joint in [6, 13]:
                current_pos[actual_joint] = max(0.0, current_pos[actual_joint] - step_size)
            else:
                current_pos[actual_joint] -= step_size
            print(f"关节 {actual_joint} 减少到: {current_pos[actual_joint]:.4f}")
            env.step(current_pos, np.array([0.2, 0.2]))
            time.sleep(0.02)
            
        # 调整步长
        if key == ord('['): # 减小步长
            step_size = max(0.001, step_size - 0.001)
            print(f"步长调整为: {step_size:.4f}")
            
        if key == ord(']'): # 增加步长
            step_size = min(0.02, step_size + 0.001)
            print(f"步长调整为: {step_size:.4f}")

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

    print("\n请设置机械臂初始位置...")
    initial_position = setup_initial_position(env)
    if initial_position is None:  # 如果按Esc退出设置模式
        thread_run = False
        cv2.destroyAllWindows()
        return
    
    # 等待确认开始运行
    print("\n按Space开始运行模型2，按Esc退出")
    cv2.namedWindow("Control")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space
            break
        if key == 27:  # Esc
            thread_run = False
            cv2.destroyAllWindows()
            return
    cv2.destroyWindow("Control")
    
    # 初始化任务参数
    episode_len = 4000
    t = 0
    current_phase = 2  # 直接从模型2开始
    first_model2 = True
    last_action = initial_position.copy()
    
    try:
        # 加载模型二
        print('\n开始加载模型二...')
        model2_name = 'policy_best.ckpt'
        model2 = Imitate_Model(ckpt_dir='./ckpt/1203_yellow_plasticbag_M_grabtoy', ckpt_name=model2_name)
        print("模型二路径：", model2.ckpt_dir)
        model2.loadModel()
        print("模型二加载成功")
        
        # 获取初始观测
        observation = {
            'qpos': initial_position,
            'images': {
                'top': image_top,
                'left_wrist': image_left,
                'right_wrist': image_right,
                'bottom': image_bottom
            }
        }
        
        # 检查观测数据
        print("\n检查观测数据...")
        if any(img is None for img in observation['images'].values()):
            raise ValueError("图像数据为空")
        print("图像数据正常")
        print("初始位置:", initial_position)
        
        # 尝试第一次预测
        print("\n尝试第一次预测...")
        try:
            action = model2.predict(observation, 0)
            # 限制夹爪状态在0-1之间
            action[6] = np.clip(initial_position[6], 0.0, 1.0)   # 左夹爪
            action[13] = np.clip(initial_position[13], 0.0, 1.0) # 右夹爪
            print("��型2第一次预测成功:", action)
        except Exception as e:
            print("模型预测失败:", str(e))
            raise
        
        # 平滑过渡阶段
        print("\n开始平滑过渡...")
        # 只计算非夹爪关节的最大增量
        non_gripper_delta = np.abs(initial_position[:6] - action[:6]).max()
        non_gripper_delta = max(non_gripper_delta, 
                              np.abs(initial_position[7:13] - action[7:13]).max())
        steps = min(int(non_gripper_delta / 0.001), 500)
        print(f"过渡步数: {steps}, 最大增量: {non_gripper_delta}")
        
        # 执行平滑过渡
        transition_positions = np.linspace(initial_position, action, steps)
        for i, jnt in enumerate(transition_positions):
            # 更新过渡点
            jnt = update_transition_point(jnt, action)
            
            # 确保过渡过程中夹爪状态不变
            jnt[6] = initial_position[6]
            jnt[13] = initial_position[13]
            if i % 50 == 0:  # 每50步打印一次进度
                print(f"过渡进度: {i}/{steps}")
            env.step(jnt, np.array([0.2, 0.2]))
            time.sleep(0.02)
        
        print("过渡结束，开始正式执行模型2...")
        
        # 开始执行模型2
        t = 0
        last_action = action.copy()
        initial_gripper_state = initial_position[[6, 13]].copy()  # 保存初始夹爪状态
        
        # 在开始执行模型2之前，保存右臂的初始位置
        right_arm_initial = initial_position[7:14].copy()  # 保存右臂所有关节位置（包括夹爪）
        
        while t < episode_len:
            try:
                # 更新观测数据
                observation = {
                    'qpos': env.get_obs()["joint_positions"],
                    'images': {
                        'top': image_top,
                        'left_wrist': image_left,
                        'right_wrist': image_right,
                        'bottom': image_bottom
                    }
                }
                
                # 模型预测
                action = model2.predict(observation, t)
                
                # 保持右臂位置不变
                action[7:14] = right_arm_initial  # 右臂所有关节（包括夹爪）保持初始位置
                
                # 确保左臂夹爪状态保持不变
                action[6] = initial_gripper_state[0]
                
                # 检查动作是否有效
                if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                    raise ValueError("模型预测包含无效值")
                
                # 行动作
                obs = env.step(action, np.array([0.6, 0.6]))
                
                # 打印调试信息
                if t % 50 == 0:  # 每50步打印一次
                    print(f"\n步数: {t}")
                    print("左臂动作增量:", action[:7] - last_action[:7])
                
                last_action = action.copy()
                t += 1
                time.sleep(0.02)
                
            except Exception as e:
                print(f"执行过程中出错: {str(e)}")
                break
                
    except Exception as e:
        print(f"程序出错: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        thread_run = False
        cv2.destroyAllWindows()
        print("��序结束")

def update_transition_point(current_position, target_position):
    # 根据当前状态和目标位置动态计算新的过渡点
    # 这里可以使用插值、平滑函数或其他策略
    new_transition_point = (current_position + target_position) / 2
    return new_transition_point

if __name__ == "__main__":
    main(tyro.cli(Args))