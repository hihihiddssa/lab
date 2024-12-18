import numpy as np
import cv2
import time
import sys
import os
import pyrealsense2 as rs

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from dobotapi.robot_control.dobot_api import DobotApiDashboard, DobotApiMove
from dobotapi.robotic_grasp.hardware.camera import RealSenseCamera
from scripts.manipulate_utils import load_ini_data_camera
from dobot_control.cameras.realsense_camera import get_device_ids

class RealSenseCamera:
    def __init__(self, device_id=None):
        self.pipeline = None
        self.config = None
        self.device_id = device_id
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            if device_id:
                self.config.enable_device(device_id)
            # 配置深度和彩色流
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        except Exception as e:
            print(f"RealSense相机初始化错误: {str(e)}")

    def start(self):
        if self.pipeline and self.config:
            try:
                self.pipeline.start(self.config)
                return True
            except Exception as e:
                print(f"相机启动错误: {str(e)}")
                return False
        return False

class EyeInHandCalibration:
    def __init__(self):
        """初始化标定系统"""
        print("正在初始化标定系统...")
        
        # 初始化机器人
        print("连接机器人...")
        try:
            self.robot_ip = "192.168.5.1"
            self.robot_dashboard = DobotApiDashboard(self.robot_ip, 29999)
            self.robot_move = DobotApiMove(self.robot_ip, 30003)
            print("机器人连接成功")
            
            # 设置机器人参数
            self.robot_dashboard.ClearError()
            self.robot_dashboard.EnableRobot()
            time.sleep(1)
            self.robot_dashboard.SpeedFactor(25)
            self.robot_dashboard.AccL(25)
            self.robot_dashboard.User(0)
            self.robot_dashboard.Tool(0)
            
        except Exception as e:
            print(f"机器人连接失败: {str(e)}")
            raise
            
        # 初始化相机
        print("连接相机...")
        try:
            # 获取相机设备ID
            device_ids = get_device_ids()
            print(f"Found {len(device_ids)} devices: ", device_ids)
            
            # 加载相机配置
            camera_dict = load_ini_data_camera()
            
            # 初始化相机列表
            self.rs_list = [
                RealSenseCamera(device_id=camera_dict["top"]),
                RealSenseCamera(device_id=camera_dict["left"]),
                RealSenseCamera(device_id=camera_dict["right"])
            ]
            print("相机初始化成功")
            
            # 标定板参数
            self.board_size = (10, 7)  # 棋盘格内角点数 (宽 × 高)
            self.square_size = 15.0   # 方格尺寸(mm)
            
            # 初始化存储列表
            self.rs_list = []  # 相机列表
            self.robot_poses = []  # 机器人位姿列表
            self.image_points = []  # 图像角点列表
            self.camera_poses = []  # 添加相机位姿列表
            
            self.generate_object_points()
            
        except Exception as e:
            print(f"相机初始化失败: {str(e)}")
            raise
            
        self.camera_matrix = np.array([[615.474, 0, 326.149],
                                     [0, 615.344, 239.753],
                                     [0, 0, 1]])  # 相机内参
        self.dist_coeffs = np.array([0.0779, -0.1301, 0, 0, 0])  # 畸变系数
        self.camera_poses = []
        
    def generate_object_points(self):
        """生成标定板角点的世界坐标"""
        self.objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        self.objp = self.objp * self.square_size
        
    def capture_calibration_data(self):
        """采集当前位姿的标定数据"""
        try:
            if not self.rs_list:
                raise Exception("未初始化相机")
                
            # 获取图像
            camera = self.rs_list[0]  # 使用第一个相机
            _img, _ = camera.read()
            if _img is None:
                raise Exception("无法获取图像")
                
            # 转换图像格式
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
            
            # 查找角点
            ret, corners = cv2.findChessboardCorners(gray, self.board_size)
            
            if ret:
                # 优化角点位置
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                
                # 获取机器人当前位姿
                robot_pose = self.robot_dashboard.GetPose()
                
                # 保存数据
                self.robot_poses.append(robot_pose)
                self.image_points.append(corners2)
                
                return True, corners2, _img
            else:
                return False, None, _img
                
        except Exception as e:
            raise Exception(f"采集标定数据失败: {str(e)}")
        
    def calculate_calibration(self):
        """计算手眼标定结果"""
        try:
            if len(self.robot_poses) < 3:
                raise Exception("数据不足，至少需要3组数据")
                
            # 准备数据
            R_gripper2base = []
            t_gripper2base = []
            R_target2cam = []
            t_target2cam = []
            
            # 处理机器人位姿数据
            for pose_str in self.robot_poses:
                pose_values = pose_str.split(',{')[1].split('}')[0].split(',')
                pose = [float(val) for val in pose_values]
                
                # 将位置从mm转换为m
                t = np.array([pose[0], pose[1], pose[2]]) / 1000.0  # 明确转换单位
                
                # 将欧拉角转换为旋转矩阵
                Rx = pose[3] * np.pi / 180.0
                Ry = pose[4] * np.pi / 180.0
                Rz = pose[5] * np.pi / 180.0
                R = self.euler_to_rotation_matrix(Rx, Ry, Rz)
                
                R_gripper2base.append(R)
                t_gripper2base.append(t.reshape(3, 1))  # 确保形状正确
            
            # 处理相机观察到的标定板数据
            for corners in self.image_points:
                ret, rvec, tvec = cv2.solvePnP(
                    self.objp / 1000.0,  # 将标定板尺寸从mm转换为m
                    corners, 
                    self.camera_matrix, 
                    self.dist_coeffs
                )
                
                R_cam, _ = cv2.Rodrigues(rvec)
                t_cam = tvec
                
                R_target2cam.append(R_cam)
                t_target2cam.append(t_cam)
            
            # 转换为numpy数组
            R_gripper2base = np.array(R_gripper2base)
            t_gripper2base = np.array(t_gripper2base)
            R_target2cam = np.array(R_target2cam)
            t_target2cam = np.array(t_target2cam)
            
            # 计算手眼标定
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # 保存结果
            self.camera_poses = {
                'R': R_cam2gripper,
                't': t_cam2gripper
            }
            
            # 保存标定结果
            self.save_calibration()
            
            # 计算标定误差
            error = self.calculate_calibration_error(
                R_cam2gripper, t_cam2gripper,
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam
            )
            
            return R_cam2gripper, t_cam2gripper, error
            
        except Exception as e:
            import traceback
            raise Exception(f"标定计算失败: {str(e)}\n{traceback.format_exc()}")
    
    def calculate_calibration_error(self, R_cam2gripper, t_cam2gripper, 
                                  R_gripper2base, t_gripper2base,
                                  R_target2cam, t_target2cam):
        """计算标定误差"""
        errors = []
        for i in range(len(R_gripper2base)):
            # 计算预测的标定板位置
            R_pred = R_gripper2base[i].dot(R_cam2gripper).dot(R_target2cam[i])
            t_pred = R_gripper2base[i].dot(R_cam2gripper).dot(t_target2cam[i]) + t_gripper2base[i]
            
            # 计算实际的标定板位置
            R_actual = R_gripper2base[i]
            t_actual = t_gripper2base[i]
            
            # 计算误差
            R_error = np.arccos((np.trace(R_pred.dot(R_actual.T)) - 1) / 2)
            t_error = np.linalg.norm(t_pred - t_actual)
            
            errors.append((R_error, t_error))
            
        return np.mean(errors, axis=0)
    
    def euler_to_rotation_matrix(self, rx, ry, rz):
        """欧拉角转旋转矩阵"""
        # 绕X轴的旋转矩阵
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # 绕Y轴的旋转矩阵
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # 绕Z轴的旋转矩阵
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # 合成旋转矩阵 R = Rz * Ry * Rx
        R = Rz.dot(Ry).dot(Rx)
        return R
        
    def save_calibration(self):
        """保存标定结果"""
        try:
            if not hasattr(self, 'camera_poses'):
                raise Exception("未进行标定")
                
            # 创建保存目录
            import os
            save_dir = "calibration_results"
            os.makedirs(save_dir, exist_ok=True)
            
            # 生成时间戳
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存标定结果到文件
            np.savez(f'{save_dir}/calibration_result_{timestamp}.npz',
                    R=self.camera_poses['R'],
                    t=self.camera_poses['t'])
            
            # 同时保存为可读文本格式
            with open(f'{save_dir}/calibration_result_{timestamp}.txt', 'w') as f:
                f.write("Eye-in-Hand Calibration Results\n")
                f.write("===============================\n\n")
                f.write("Rotation Matrix (R):\n")
                f.write(str(self.camera_poses['R']))
                f.write("\n\nTranslation Vector (t) [meters]:\n")
                f.write(str(self.camera_poses['t']))
                f.write("\n\nEuler Angles [degrees]:\n")
                euler = self.rotation_matrix_to_euler(self.camera_poses['R'])
                f.write(f"Rx: {euler[0]:.3f}\n")
                f.write(f"Ry: {euler[1]:.3f}\n")
                f.write(f"Rz: {euler[2]:.3f}\n")
                
            return f'{save_dir}/calibration_result_{timestamp}'
                
        except Exception as e:
            raise Exception(f"保存标定结果失败: {str(e)}")
            
    def rotation_matrix_to_euler(self, R):
        """旋转矩阵转欧拉角（度数）"""
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z]) * 180.0 / np.pi  # 转换为度数

    def initialize_cameras(self):
        """初始化相机"""
        try:
            print("连接相机...")
            if self.camera and self.camera.start():
                print("相机初始化成功")
                return True
            else:
                print("相机启动失败")
                return False
        except Exception as e:
            print(f"相机初始化错误: {str(e)}")
            return False
