import os
import numpy as np
import json
import cv2
import time
import torch
import torch.nn as nn
from dobot_api import DobotApiDashboard, DobotApiMove
import pyrealsense2 as rs
import logging
import traceback

from dobotapi.robotic_grasp.utils.data.camera_data import current_dir
from hardware.camera import RealSenseCamera
from hardware.device import get_device
from inference.post_process import post_process_output
from utils.data.camera_data import CameraData

logging.basicConfig(level=logging.INFO)

class GraspingControl:
    def __init__(self, model_path):
        # 机器人IP和端口配置
        self.ip = "192.168.5.1"
        self.dashboard_port = 29999
        self.move_port = 30003

        # 初始化机器人控制
        self.robot_dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
        self.robot_move = DobotApiMove(self.ip, self.move_port)

        # 安全限位参数
        self.MIN_Z_HEIGHT = 35.0
        self.SAFE_X_RANGE = (-410, 210)  # 左臂X轴范围
        self.SAFE_Y_RANGE = (-700, -210)  # Y轴范围

        # 加载标定数据
        self.load_calibration_data()

        # 初始化相机
        self.init_camera()

        # 初始化机器人
        self.init_robot()

        # 加载神经网络模型
        self.load_model(model_path)

    def load_model(self, model_path):
        """加载预训练的神经网络模型"""
        logging.info('Loading model...')
        self.device = get_device(False)  # 使用与 run_realtime.py 相同的设备获取方式
        self.net = torch.load(model_path, map_location=self.device)
        self.net.eval()
        logging.info('Done')

    def init_camera(self):
        """初始化相机"""
        try:
            # 初始化RealSense相机
            ctx = rs.context()
            device_id = '130322273294'  # 使用固定的设备ID
            self.camera = RealSenseCamera(
                device_id=device_id,
                width=640,
                height=480,
                fps=30
            )
            time.sleep(2)  # 等待相机初始化
            self.camera.connect()
            self.cam_data = CameraData(include_depth=True, include_rgb=True)

            # 加载相机内参
            self.load_camera_intrinsics()

            print("相机初始化成功")
        except Exception as e:
            print(f"相机初始化失败: {str(e)}")
            raise

    def load_camera_intrinsics(self):
        """加载相机内参"""
        # 这里使用RealSense相机的实际内参
        self.camera_matrix = np.array([
            [615.7725830078125, 0.0, 324.38262939453125],
            [0.0, 615.9288940429688, 237.60375976562],
            [0.0, 0.0, 1.0]
        ])
        self.dist_coeffs = np.zeros((5,1))  # 假设无畸变

    def init_robot(self):
        """初始化机器人"""
        try:
            # 清除错误并使能机器人
            self.robot_dashboard.ClearError()
            self.robot_dashboard.EnableRobot()
            time.sleep(1)

            # 设置速度和加速度
            self.robot_dashboard.SpeedFactor(25)  # 设置速度为25%
            self.robot_dashboard.AccL(25)
            self.robot_dashboard.SpeedL(25)

            # 设置工具和用户坐标系
            self.robot_dashboard.User(0)
            self.robot_dashboard.Tool(0)

            print("机器人初始化成功")
        except Exception as e:
            print(f"机器人初始化失败: {str(e)}")
            raise

    def load_calibration_data(self):
        """加载手眼标定数据"""
        try:
            current_dir = os.getcwd()
            print('当前目录：', current_dir)
            with open('calibration_result.txt', 'r') as f:
                lines = f.readlines()

                # 解析旋转矩阵
                R_start = lines.index("Rotation Matrix (R):\n") + 1
                R = []
                for i in range(R_start, R_start + 3):
                    R.append([float(x) for x in lines[i].strip().replace('[', '').replace(']', '').split()])
                self.R = np.array(R)

                # 解析平移向量
                t_start = lines.index("Translation Vector (t) [meters]:\n") + 1
                t = []
                for i in range(t_start, t_start + 3):
                    t.append(float(lines[i].strip().replace('[', '').replace(']', '')))
                self.t = np.array(t).reshape(3, 1)

            print("标定数据加载成功")
        except Exception as e:
            print(f"加载标定数据失败: {str(e)}")
            raise

    def pixel_to_camera(self, x, y, depth):
        """
        将像素坐标转换为相机坐标系下的3D点
        x, y: 像素坐标
        depth: 深度值(米)
        返回: 相机坐标系下的3D点 [X, Y, Z]
        """
        # 使用相机内参矩阵的逆矩阵将像素坐标转换为归一化坐标
        fx = self.camera_matrix[0,0]
        fy = self.camera_matrix[1,1]
        cx = self.camera_matrix[0,2]
        cy = self.camera_matrix[1,2]

        # 计算相机坐标
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    def transform_camera_to_robot(self, camera_point):
        """
        将相机坐标系下的点转换到机器人坐标系
        camera_point: [x, y, z] 在相机坐标系下的坐标
        """
        # 确保camera_point是正确的形状
        camera_point = np.array(camera_point).reshape(3, 1)
        
        # 使用标定数据进行转换
        robot_point = self.R @ camera_point + self.t
        return robot_point.flatten()

    def check_position_safety(self, position):
        """
        检查位置是否在安全范围内
        position: [x, y, z, rx, ry, rz]
        返回: bool
        """
        x, y, z = position[:3]

        # 检查Z轴高度
        if z < self.MIN_Z_HEIGHT:
            print(f"警告: Z轴高度 ({z:.2f}) 低于安全高度")
            return False

        # 检查XY范围
        if not (self.SAFE_X_RANGE[0] <= x <= self.SAFE_X_RANGE[1]):
            print(f"警告: X轴位置 ({x:.2f}) 超出安全范围")
            return False

        if not (self.SAFE_Y_RANGE[0] <= y <= self.SAFE_Y_RANGE[1]):
            print(f"警告: Y轴位置 ({y:.2f}) 超出安全范围")
            return False

        return True

    def get_grasp_pose(self, q_img, ang_img, depth_img):
        """
        从预测结果中获取最佳抓取位置
        返回: (camera_point, angle)
        """
        try:
            # 检查并转换数据类型
            if torch.is_tensor(q_img):
                q_img = q_img.detach().cpu().numpy()
            if torch.is_tensor(ang_img):
                ang_img = ang_img.detach().cpu().numpy()
            
            # 找到质量图中的最大值位置
            max_pos = np.unravel_index(np.argmax(q_img), q_img.shape)
            y, x = max_pos
            
            # 直接使用标定数据中的旋转矩阵和平移向量
            pixel_point = np.array([x, y, 1.0]).reshape(3, 1)
            
            # 使用相机内参矩阵和标定数据计算3D点
            camera_point = np.linalg.inv(self.camera_matrix) @ pixel_point
            camera_point = camera_point.flatten()
            
            # 使用标定数据转换到机器人坐标系
            robot_point = self.R @ camera_point.reshape(3,1) + self.t
            
            angle = ang_img[y, x]
            if isinstance(angle, np.ndarray):
                angle = angle.item()

            print(f"抓取位置: x={x}, y={y}, angle={angle}")
            print(f"相机坐标: {camera_point}")
            print(f"机器人坐标: {robot_point.flatten()}")

            return camera_point, angle

        except Exception as e:
            print(f"获取抓取姿态时出错: {str(e)}")
            print(f"错误详情:\n{traceback.format_exc()}")
            return None, None

    def execute_grasp(self, q_img, ang_img, depth_img):
        """
        执行抓取动作
        """
        try:
            # 1. 获取抓取位置和角度
            camera_point, angle = self.get_grasp_pose(q_img, ang_img, depth_img)
            
            if camera_point is None or angle is None:
                print("无法获取有效的抓取位置")
                return False
            
            print(f"相机坐标系下的抓取点: {camera_point}")

            # 2. 将相机坐标转换为机器人坐标
            robot_point = self.transform_camera_to_robot(camera_point)
            print(f"机器人坐标系下的抓取点: {robot_point}")

            # 3. 构建完整的机器人位姿
            robot_pose = [
                float(robot_point[0]),  # X
                float(robot_point[1]),  # Y
                float(robot_point[2]),  # Z
                0.0,                    # Rx
                0.0,                    # Ry
                float(angle)            # Rz (使用预的抓取角度)
            ]

            # 4. 检查位置安全性
            if not self.check_position_safety(robot_pose):
                print("抓取位置不安全，放弃执行")
                return False

            # 5. 控制机器人移动到抓取位置
            print(f"移动到抓取位置: {robot_pose}")
            self.robot_move.MovL(*robot_pose)

            return True

        except Exception as e:
            print(f"执行抓取失败: {str(e)}")
            print(f"错误详情:\n{traceback.format_exc()}")
            return False

    def run_grasp_detection(self):
        """运行抓取检测主循环"""
        try:
            # 创建主窗口
            cv2.namedWindow('Grasp Detection')
            
            # 创建控制窗口
            cv2.namedWindow('Controls')
            
            # 创建ROI大小的滑动条
            cv2.createTrackbar('ROI Size', 'Controls', 50, 200, lambda x: None)
            
            # 创建执行按钮
            def execute_callback(state):
                if self.roi is not None:
                    print("\n准备执行抓取...")
                    self.execute_flag = True
                else:
                    print("\n请先选择抓取区域")
                    
            cv2.createButton('Execute Grasp', execute_callback, None, cv2.QT_PUSH_BUTTON)
            
            # 用于存储ROI选择
            self.roi = None
            self.execute_flag = False
            self.grasp_info = None  # 存储抓取信息
            
            def mouse_callback(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    self.roi = [x, y]
                    print(f"\n已选择点: ({x}, {y})")
            
            cv2.setMouseCallback('Grasp Detection', mouse_callback)

            while True:
                # 获取相机图像
                image_bundle = self.camera.get_image_bundle()
                rgb = image_bundle['rgb']
                depth = image_bundle['aligned_depth']

                # 处理图像数据
                x, depth_img, rgb_img = self.cam_data.get_data(rgb=rgb, depth=depth)
                
                # 添加batch维度
                x = x.unsqueeze(0)

                # 进行抓取预测
                with torch.no_grad():
                    xc = x.to(self.device)
                    pred = self.net.predict(xc)

                    q_img, ang_img, width_img = post_process_output(
                        pred['pos'], pred['cos'], pred['sin'], pred['width']
                    )

                    # 如果有ROI选择，更新q_img
                    if self.roi is not None:
                        # 获取当前ROI大小
                        roi_size = cv2.getTrackbarPos('ROI Size', 'Controls')
                        
                        # 创建掩码
                        mask = np.zeros_like(q_img)
                        x1 = max(0, self.roi[0] - roi_size)
                        x2 = min(q_img.shape[1], self.roi[0] + roi_size)
                        y1 = max(0, self.roi[1] - roi_size)
                        y2 = min(q_img.shape[0], self.roi[1] + roi_size)
                        mask[y1:y2, x1:x2] = 1
                        q_img = q_img * mask

                    # 获取抓取信息
                    camera_point, angle = self.get_grasp_pose(q_img, ang_img, depth_img)
                    if camera_point is not None:
                        robot_point = self.transform_camera_to_robot(camera_point)
                        self.grasp_info = {
                            'camera_point': camera_point,
                            'robot_point': robot_point,
                            'angle': angle
                        }

                    # 显示结果
                    self.visualize_results(rgb_img, depth_img, q_img, ang_img, width_img)

                    # 如果执行标志被设置
                    if self.execute_flag:
                        try:
                            if self.execute_grasp(q_img, ang_img, depth_img):
                                print("\n抓取执行成功")
                            else:
                                print("\n抓取执行失败")
                        except Exception as e:
                            print(f"\n执行抓取时出错: {str(e)}")
                        finally:
                            self.execute_flag = False

                # 检查键盘输入
                key = cv2.waitKey(1)
                if key == 27:  # ESC键退出
                    break
                elif key == ord('c'):  # 'c'键清除ROI选择
                    self.roi = None
                    print("\n已清除选择区域")

        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.cleanup()

    def visualize_results(self, rgb_img, depth_img, q_img, ang_img, width_img=None):
        """显示检测结果，包含抓取框的可视化"""
        try:
            # 检查并转换数据类型
            if torch.is_tensor(rgb_img):
                rgb_img = rgb_img.detach().cpu().numpy()
            if torch.is_tensor(q_img):
                q_img = q_img.detach().cpu().numpy()
            if torch.is_tensor(ang_img):
                ang_img = ang_img.detach().cpu().numpy()
            
            # 确保图像数据类型和格式正确
            if rgb_img.dtype != np.uint8:
                if rgb_img.max() <= 1.0:
                    rgb_img = (rgb_img * 255).astype(np.uint8)
                else:
                    rgb_img = rgb_img.astype(np.uint8)
            
            # 确保图像是3通道的
            if len(rgb_img.shape) == 2:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_GRAY2BGR)
            elif len(rgb_img.shape) == 3 and rgb_img.shape[2] == 4:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGR)
            elif len(rgb_img.shape) == 3 and rgb_img.shape[2] != 3:
                rgb_img = rgb_img.transpose(1, 2, 0)
                if rgb_img.shape[2] == 4:
                    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGBA2BGR)
            
            # 创建可视化图像
            vis_img = rgb_img.copy()
            
            # 找到最佳抓取点
            max_pos = np.unravel_index(np.argmax(q_img), q_img.shape)
            y, x = max_pos
            
            # 确保坐标在有效范围内
            y = min(y, vis_img.shape[0]-1)
            x = min(x, vis_img.shape[1]-1)
            
            # 获取抓取角度
            angle = ang_img[y, x]
            
            # 绘制抓取框
            GRIPPER_GRASP_LENGTH = 60
            GRIPPER_GRASP_WIDTH = 20
            
            # 计算抓取框的四个角点
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            
            # 计算抓取框的四个角点
            corner_points = np.array([
                [-GRIPPER_GRASP_WIDTH/2, -GRIPPER_GRASP_LENGTH/2],
                [GRIPPER_GRASP_WIDTH/2, -GRIPPER_GRASP_LENGTH/2],
                [GRIPPER_GRASP_WIDTH/2, GRIPPER_GRASP_LENGTH/2],
                [-GRIPPER_GRASP_WIDTH/2, GRIPPER_GRASP_LENGTH/2]
            ])
            
            # 旋转角点
            rotation_matrix = np.array([
                [cos_theta, -sin_theta],
                [sin_theta, cos_theta]
            ])
            rotated_corners = np.dot(corner_points, rotation_matrix.T)
            
            # 移动到抓取中心点
            corners = rotated_corners + np.array([x, y])
            corners = corners.astype(np.int32)
            
            # 绘制抓取框
            cv2.polylines(vis_img, [corners], True, (0, 255, 0), 2)
            
            # 绘制抓取点
            cv2.circle(vis_img, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # 如果有ROI选择，绘制ROI
            if self.roi is not None:
                roi_size = cv2.getTrackbarPos('ROI Size', 'Controls')
                cv2.circle(vis_img, (self.roi[0], self.roi[1]), 5, (255, 0, 0), -1)
                cv2.rectangle(vis_img, 
                             (self.roi[0] - roi_size, self.roi[1] - roi_size),
                             (self.roi[0] + roi_size, self.roi[1] + roi_size),
                             (255, 0, 0), 2)
            
            # 显示抓取信息
            if self.grasp_info is not None:
                info_text = [
                    f"Camera XYZ: {self.grasp_info['camera_point']}",
                    f"Robot XYZ: {self.grasp_info['robot_point']}",
                    f"Angle: {self.grasp_info['angle']:.2f}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(vis_img, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 显示图像
            cv2.imshow("Grasp Detection", vis_img)
            
        except Exception as e:
            print(f"\n可视化结果时出错: {str(e)}")
            print(f"错误详情: ", traceback.format_exc())

    def cleanup(self):
        """清理资源"""
        try:
            # 关闭相机
            if hasattr(self, 'camera'):
                self.camera.pipeline.stop()  # 使用pipeline.stop()替代disconnect()

            # 关闭机器人连接
            if hasattr(self, 'robot_dashboard'):
                self.robot_dashboard.DisableRobot()
                self.robot_dashboard.close()
            if hasattr(self, 'robot_move'):
                self.robot_move.close()

            # 关闭所有窗口
            cv2.destroyAllWindows()

        except Exception as e:
            print(f"清理资源时出错: {str(e)}")

def main():
    try:
        # 创建抓取控制器
        model_path = 'trained-models/cornell-randsplit-rgbd-grconvnet3-drop1-ch16/epoch_30_iou_0.97'
        controller = GraspingControl(model_path)

        # 运行抓取检测
        controller.run_grasp_detection()

    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
