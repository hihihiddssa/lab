import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from biaoding import EyeInHandCalibration

class CalibrationGUI:
    def __init__(self):
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("手眼标定系统")
        
        # 创建标定对象
        print("初始化标定系统...")
        self.calibrator = EyeInHandCalibration()
        
        # 设置窗口大小
        self.root.geometry("1920x1080")
        
        # 初始化显示画布
        self.show_canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # 修改为单相机尺寸
        
        self.setup_ui()
        self.initialize_robot()
        self.initialize_cameras()
        
    def setup_ui(self):
        # 创建左侧控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side="left", padx=10, pady=10, fill="y")
        
        # 机器人控制区域
        robot_frame = ttk.LabelFrame(control_frame, text="机器人控制")
        robot_frame.pack(fill="x", padx=5, pady=5)
        
        # 位置输�����框
        pos_frame = ttk.Frame(robot_frame)
        pos_frame.pack(padx=5, pady=5)
        
        self.pos_entries = []
        labels = ['X:', 'Y:', 'Z:', 'Rx:', 'Ry:', 'Rz:']
        for i, label in enumerate(labels):
            ttk.Label(pos_frame, text=label).grid(row=i//3, column=(i%3)*2, padx=2)
            entry = ttk.Entry(pos_frame, width=8)
            entry.grid(row=i//3, column=(i%3)*2+1, padx=2, pady=2)
            self.pos_entries.append(entry)
            
        # 机器人控制按钮
        btn_frame = ttk.Frame(robot_frame)
        btn_frame.pack(pady=5)
        
        ttk.Button(btn_frame, text="移动到位置", 
                  command=self.move_to_position).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="获取当前位置", 
                  command=self.get_current_position).pack(side="left", padx=5)
        
        # 添加机器人控制面板
        robot_control_frame = ttk.LabelFrame(control_frame, text="机器人运动控制")
        robot_control_frame.pack(fill="x", padx=5, pady=5)
        
        # 步进值设置
        step_frame = ttk.Frame(robot_control_frame)
        step_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(step_frame, text="步进值(mm/°):").pack(side="left", padx=5)
        self.step_size = ttk.Entry(step_frame, width=8)
        self.step_size.pack(side="left", padx=5)
        self.step_size.insert(0, "10")  # 默认步进值
        
        # 创建方向控制按钮
        btn_frame = ttk.Frame(robot_control_frame)
        btn_frame.pack(pady=5)
        
        # X/Y方向控制
        xy_frame = ttk.Frame(btn_frame)
        xy_frame.pack(pady=5)
        
        ttk.Button(xy_frame, text="Y+", width=5,
                  command=lambda: self.move_step(0, 1, 0, 0, 0, 0)).grid(row=0, column=1)
        ttk.Button(xy_frame, text="X-", width=5,
                  command=lambda: self.move_step(-1, 0, 0, 0, 0, 0)).grid(row=1, column=0)
        ttk.Button(xy_frame, text="X+", width=5,
                  command=lambda: self.move_step(1, 0, 0, 0, 0, 0)).grid(row=1, column=2)
        ttk.Button(xy_frame, text="Y-", width=5,
                  command=lambda: self.move_step(0, -1, 0, 0, 0, 0)).grid(row=2, column=1)
        
        # Z方向和旋转控制
        z_rot_frame = ttk.Frame(btn_frame)
        z_rot_frame.pack(pady=5)
        
        ttk.Button(z_rot_frame, text="Z+", width=5,
                  command=lambda: self.move_step(0, 0, 1, 0, 0, 0)).grid(row=0, column=0)
        ttk.Button(z_rot_frame, text="Z-", width=5,
                  command=lambda: self.move_step(0, 0, -1, 0, 0, 0)).grid(row=0, column=1)
        
        ttk.Button(z_rot_frame, text="Rx+", width=5,
                  command=lambda: self.move_step(0, 0, 0, 1, 0, 0)).grid(row=1, column=0)
        ttk.Button(z_rot_frame, text="Rx-", width=5,
                  command=lambda: self.move_step(0, 0, 0, -1, 0, 0)).grid(row=1, column=1)
        
        ttk.Button(z_rot_frame, text="Ry+", width=5,
                  command=lambda: self.move_step(0, 0, 0, 0, 1, 0)).grid(row=2, column=0)
        ttk.Button(z_rot_frame, text="Ry-", width=5,
                  command=lambda: self.move_step(0, 0, 0, 0, -1, 0)).grid(row=2, column=1)
        
        ttk.Button(z_rot_frame, text="Rz+", width=5,
                  command=lambda: self.move_step(0, 0, 0, 0, 0, 1)).grid(row=3, column=0)
        ttk.Button(z_rot_frame, text="Rz-", width=5,
                  command=lambda: self.move_step(0, 0, 0, 0, 0, -1)).grid(row=3, column=1)
        
        # 速度控制
        speed_frame = ttk.Frame(robot_control_frame)
        speed_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(speed_frame, text="速度(%):").pack(side="left", padx=5)
        self.speed_value = ttk.Entry(speed_frame, width=8)
        self.speed_value.pack(side="left", padx=5)
        self.speed_value.insert(0, "20")  # 默认速度
        
        ttk.Button(speed_frame, text="设置速度",
                  command=self.set_robot_speed).pack(side="left", padx=5)
        
        # 标定控制区域
        calib_frame = ttk.LabelFrame(control_frame, text="标定控制")
        calib_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(calib_frame, text="采集当前位姿", 
                  command=self.capture_current_pose).pack(pady=5)
        
        # 添加保存和加载按钮
        save_load_frame = ttk.Frame(calib_frame)
        save_load_frame.pack(fill="x", pady=5)
        
        ttk.Button(save_load_frame, text="保存数据", 
                  command=self.save_calibration_data).pack(side="left", padx=5)
        ttk.Button(save_load_frame, text="加载数据", 
                  command=self.load_calibration_data).pack(side="left", padx=5)
        
        self.data_count_label = ttk.Label(calib_frame, text="已采集数据: 0 组")
        self.data_count_label.pack(pady=5)
        
        ttk.Button(calib_frame, text="计算标定结果", 
                  command=self.calculate_calibration).pack(pady=5)
        
        # 在标定控制区域添加自动采集
        auto_frame = ttk.LabelFrame(calib_frame, text="自动采集设置")
        auto_frame.pack(fill="x", pady=5)
        
        # 采集点数设置
        points_frame = ttk.Frame(auto_frame)
        points_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(points_frame, text="采集点数:").pack(side="left", padx=5)
        self.points_count = ttk.Entry(points_frame, width=8)
        self.points_count.pack(side="left", padx=5)
        self.points_count.insert(0, "12")  # 默认12个点
        
        # 自动采集按钮
        ttk.Button(auto_frame, text="开始自动采集",
                  command=self.start_auto_capture).pack(pady=5)
        
        # 状态显示区域
        status_frame = ttk.LabelFrame(control_frame, text="状态信息")
        status_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(status_frame, height=10, width=40)
        self.status_text.pack(padx=5, pady=5, fill="both", expand=True)
        
        # 修改图像显示区域
        image_frame = ttk.LabelFrame(self.root, text="相机图像")
        image_frame.pack(side="right", padx=10, pady=10, fill="both", expand=True)
        
        # 创建图像显示区域
        self.image_display = tk.Canvas(image_frame, width=640*3, height=480)
        self.image_display.pack(padx=5, pady=5)
        
        # 在控制面板添加相机重连按钮
        camera_frame = ttk.LabelFrame(control_frame, text="相机控制")
        camera_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(camera_frame, text="重新连接相机", 
                  command=self.initialize_cameras).pack(pady=5)
        
        # 添加标定板参数设置区域
        board_frame = ttk.LabelFrame(control_frame, text="标定板参数")
        board_frame.pack(fill="x", padx=5, pady=5)
        
        # 棋盘格尺寸设置
        size_frame = ttk.Frame(board_frame)
        size_frame.pack(padx=5, pady=5)
        
        ttk.Label(size_frame, text="棋盘格内角点数:").grid(row=0, column=0, padx=2)
        self.board_width = ttk.Entry(size_frame, width=5)
        self.board_width.grid(row=0, column=1, padx=2)
        self.board_width.insert(0, str(self.calibrator.board_size[0]))
        
        ttk.Label(size_frame, text="×").grid(row=0, column=2, padx=2)
        
        self.board_height = ttk.Entry(size_frame, width=5)
        self.board_height.grid(row=0, column=3, padx=2)
        self.board_height.insert(0, str(self.calibrator.board_size[1]))
        
        # 方格尺寸设置
        ttk.Label(size_frame, text="方格尺寸(mm):").grid(row=1, column=0, padx=2, pady=5)
        self.square_size = ttk.Entry(size_frame, width=8)
        self.square_size.grid(row=1, column=1, columnspan=2, padx=2, pady=5)
        self.square_size.insert(0, str(self.calibrator.square_size))
        
        # 更新按钮
        ttk.Button(board_frame, text="更新标定板参数", 
                  command=self.update_board_params).pack(pady=5)
        
        # 添加数据显示区域
        data_frame = ttk.LabelFrame(control_frame, text="采集数据列表")
        data_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 创建数据显示列表
        self.data_tree = ttk.Treeview(data_frame, columns=("ID", "X", "Y", "Z", "Rx", "Ry", "Rz"), 
                                     show="headings", height=10)
        
        # 设置列标题
        self.data_tree.heading("ID", text="#")
        self.data_tree.heading("X", text="X")
        self.data_tree.heading("Y", text="Y")
        self.data_tree.heading("Z", text="Z")
        self.data_tree.heading("Rx", text="Rx")
        self.data_tree.heading("Ry", text="Ry")
        self.data_tree.heading("Rz", text="Rz")
        
        # 设置列宽
        self.data_tree.column("ID", width=30)
        for col in ("X", "Y", "Z", "Rx", "Ry", "Rz"):
            self.data_tree.column(col, width=70)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(data_frame, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)
        
        # 放置列表和滚动条
        self.data_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 添加删除按钮
        btn_frame = ttk.Frame(data_frame)
        btn_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(btn_frame, text="删除选中", 
                  command=self.delete_selected_data).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="清空数据", 
                  command=self.clear_all_data).pack(side="left", padx=5)
        
        # 在标定控制区域添加验证按钮
        ttk.Button(calib_frame, text="验证标定结果", 
                  command=self.verify_calibration).pack(pady=5)
        
    def initialize_robot(self):
        """初始化机器人"""
        try:
            self.calibrator.robot_dashboard.EnableRobot()
            self.calibrator.robot_dashboard.ClearError()
            time.sleep(1)
            
            # 设置运动参数
            self.calibrator.robot_dashboard.SpeedJ(20)
            self.calibrator.robot_dashboard.SpeedL(20)
            self.calibrator.robot_dashboard.AccJ(20)
            self.calibrator.robot_dashboard.AccL(20)
            
            self.log("机器人初始化成功")
        except Exception as e:
            self.log(f"机器人初始化失败: {str(e)}")
            
    def initialize_cameras(self):
        """初始化相机"""
        try:
            # 先尝试释放现有相机资源
            if hasattr(self.calibrator, 'rs_list') and self.calibrator.rs_list:
                for camera in self.calibrator.rs_list:
                    if camera and camera.pipeline:
                        try:
                            camera.pipeline.stop()
                        except:
                            pass
                self.calibrator.rs_list = []
            
            self.log("开始初始化相机...")
            
            # 初始化指定相机
            from dobot_control.cameras.realsense_camera import RealSenseCamera
            try:
                camera = RealSenseCamera(flip=True, device_id="130322273294")
                self.calibrator.rs_list = [camera]
                self.log("相机初始化完成")
                
                # 测试相机
                try:
                    _img, _ = camera.read()
                    if _img is not None:
                        self.log("相机测试成功")
                    else:
                        self.log("相机无法读取图像")
                except Exception as e:
                    self.log(f"相机测试失败: {str(e)}")
                    
                return True
                
            except Exception as e:
                self.log(f"相机初始化失败: {str(e)}")
                return False
            
        except Exception as e:
            self.log(f"相机初始化过程出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            return False
        
    def update_image(self):
        """更新相机图像显示"""
        try:
            # 检查相机列表是否为空
            if not self.calibrator.rs_list:
                self.log("错误: 未检测到相机连接")
                self.show_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                return
                
            # 更新相机图像
            camera = self.calibrator.rs_list[0]
            try:
                _img, _ = camera.read()
                if _img is None:
                    raise Exception("无法读取图像")
                    
                # 确保图像是正确的格式和类型
                _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
                _img = np.asarray(_img, dtype=np.uint8)
                
                # 查找棋盘格角点
                gray = cv2.cvtColor(_img, cv2.COLOR_RGB2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.calibrator.board_size)
                
                # 创建图像副本用于绘制
                display_img = _img.copy()
                
                # 绘制角点
                if ret:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    cv2.drawChessboardCorners(display_img, self.calibrator.board_size, corners2, ret)
                    cv2.putText(display_img, "Board Detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "No Board", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 更新显示画布
                self.show_canvas = display_img
                
            except Exception as e:
                self.log(f"相机获取图像失败: {str(e)}")
                self.show_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                error_img = self.show_canvas.copy()
                cv2.putText(error_img, "Camera Error", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.show_canvas = error_img
            
            # 转换为tkinter图像并显示
            try:
                image = Image.fromarray(self.show_canvas)
                photo = ImageTk.PhotoImage(image=image)
                self.image_display.delete("all")
                self.image_display.create_image(0, 0, anchor="nw", image=photo)
                self.image_display.image = photo
            except Exception as e:
                self.log(f"图像显示失败: {str(e)}")
            
        except Exception as e:
            self.log(f"更新图像失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
        finally:
            self.root.after(100, self.update_image)
        
    def move_to_position(self):
        """移动到指定位置"""
        try:
            # 获取输入框中的值
            pos = [float(entry.get()) for entry in self.pos_entries]
            self.calibrator.robot_move.MovJ(*pos)
            self.log(f"移动到位置: {pos}")
        except Exception as e:
            self.log(f"移动失败: {str(e)}")
            
    def get_current_position(self):
        """获取当前位置"""
        try:
            # GetPose()返回的是类似 "0,{-118.141403,-344.319641,415.981628,-179.962280,0.054863,89.871269}" 的格式
            pose_str = self.calibrator.robot_dashboard.GetPose()
            
            # 解析返回的字符串
            # 提取花括号中的数值部分
            pose_values = pose_str.split(',{')[1].split('}')[0].split(',')
            pose = [float(val) for val in pose_values]
            
            # 更新输入框
            for entry, value in zip(self.pos_entries, pose):
                entry.delete(0, tk.END)
                entry.insert(0, "{:.3f}".format(value))
            
            self.log("当前位置: " + str(pose))
            
        except Exception as e:
            self.log("获取位置失败: " + str(e))
            
    def capture_current_pose(self):
        """采集当前位姿的标定数据"""
        try:
            ret, corners, img = self.calibrator.capture_calibration_data()
            if ret:
                # 获取当前位姿并更新显示
                pose_str = self.calibrator.robot_dashboard.GetPose()
                pose_values = pose_str.split(',{')[1].split('}')[0].split(',')
                pose = [float(val) for val in pose_values]
                
                # 添加到树形列表
                count = len(self.calibrator.robot_poses)
                self.data_tree.insert("", "end", values=(
                    count,
                    "{:.3f}".format(pose[0]),
                    "{:.3f}".format(pose[1]),
                    "{:.3f}".format(pose[2]),
                    "{:.3f}".format(pose[3]),
                    "{:.3f}".format(pose[4]),
                    "{:.3f}".format(pose[5])
                ))
                
                self.data_count_label.config(text=f"已采集数据: {count} 组")
                self.log("数据采集成功")
            else:
                self.log("未检测到标定板")
        except Exception as e:
            self.log(f"数据采集失败: {str(e)}")
            
    def calculate_calibration(self):
        """计算标定结果"""
        try:
            if len(self.calibrator.robot_poses) < 3:
                self.log("数据不足，至少需要3组数据")
                return
                
            R, t, error = self.calibrator.calculate_calibration()
            
            # 将结果保存到标定器对象中
            self.calibrator.R = R
            self.calibrator.t = t
            self.calibrator.calibration_error = error
            
            # 显示结果
            self.log("\n标定成功!")
            self.log("\n旋转矩阵R:")
            self.log(str(R))
            self.log("\n平移向量t (米):")
            self.log(str(t))
            self.log("\n标定误差:")
            self.log(f"旋转误差: {error[0]:.6f} 弧度")
            self.log(f"平移误差: {error[1]:.6f} 米")
            
            # 显示欧拉角
            euler = self.calibrator.rotation_matrix_to_euler(R)
            self.log("\n欧拉角 (度):")
            self.log(f"Rx: {euler[0]:.3f}")
            self.log(f"Ry: {euler[1]:.3f}")
            self.log(f"Rz: {euler[2]:.3f}")
            
            # 显示保存位置
            save_path = self.calibrator.save_calibration()
            self.log(f"\n标定结果已保存到: {save_path}")
            
        except Exception as e:
            self.log(f"标定计算失败: {str(e)}")
            
    def log(self, message):
        """显示日志信息"""
        self.status_text.insert(tk.END, str(message) + "\n")
        self.status_text.see(tk.END)
        
    def run(self):
        """运行程序"""
        # 开始图像更新
        self.update_image()
        # 运行主循环
        self.root.mainloop()
        
    def update_board_params(self):
        """更新标定板参数"""
        try:
            # 获取输入值
            width = int(self.board_width.get())
            height = int(self.board_height.get())
            square_size = float(self.square_size.get())
            
            # 更新标定器参数
            self.calibrator.board_size = (width, height)
            self.calibrator.square_size = square_size
            
            # 重新生成标定板角点的世界坐标
            self.calibrator.generate_object_points()
            
            self.log(f"标定板参数已更新: {width}×{height}, 方格尺寸: {square_size}mm")
            
        except ValueError as e:
            self.log("请输入有效的数值")
        except Exception as e:
            self.log(f"更新标定板参数失败: {str(e)}")
        
    def delete_selected_data(self):
        """删除选中的数据"""
        try:
            selected_items = self.data_tree.selection()
            if not selected_items:
                self.log("请先选择要删除的数据")
                return
            
            # 获取所有选中项
            to_delete = []
            for item in selected_items:
                to_delete.append(item)
                
            # 从树形列表中删除
            for item in to_delete:
                self.data_tree.delete(item)
                
            # 更新标定器中的数据列表
            remaining_data = []
            remaining_points = []
            
            # 遍历树形列表中剩余的数据
            for i, item in enumerate(self.data_tree.get_children()):
                values = self.data_tree.item(item)["values"]
                pose_str = self.calibrator.robot_poses[i]
                remaining_data.append(pose_str)
                remaining_points.append(self.calibrator.image_points[i])
                
            # 更新数据
            self.calibrator.robot_poses = remaining_data
            self.calibrator.image_points = remaining_points
            
            # 更新显示
            self.update_data_display()
            self.log(f"已删除 {len(to_delete)} 组数据")
            
        except Exception as e:
            self.log(f"删除数据失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def clear_all_data(self):
        """清空所有数据"""
        try:
            if not self.calibrator.robot_poses:
                self.log("没有数据需要清空")
                return
            
            # 清空数据
            self.calibrator.robot_poses.clear()
            self.calibrator.image_points.clear()
            
            # 更新显示
            self.update_data_display()
            self.log("已清空所有数据")
            
        except Exception as e:
            self.log(f"清空数据失败: {str(e)}")

    def update_data_display(self):
        """更新数据显示"""
        try:
            # 清空现有显示
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # 重新显示所有数据
            for i, pose_str in enumerate(self.calibrator.robot_poses):
                pose_values = pose_str.split(',{')[1].split('}')[0].split(',')
                pose = [float(val) for val in pose_values]
                
                self.data_tree.insert("", "end", values=(
                    i,
                    "{:.3f}".format(pose[0]),
                    "{:.3f}".format(pose[1]),
                    "{:.3f}".format(pose[2]),
                    "{:.3f}".format(pose[3]),
                    "{:.3f}".format(pose[4]),
                    "{:.3f}".format(pose[5])
                ))
            
            # 更新计数
            count = len(self.calibrator.robot_poses)
            self.data_count_label.config(text=f"已采集数据: {count} 组")
            
        except Exception as e:
            self.log(f"更新数据显示失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
        
    def move_step(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0):
        """按步进值移动机器人"""
        try:
            # 获取当前位置
            pose_str = self.calibrator.robot_dashboard.GetPose()
            pose_values = pose_str.split(',{')[1].split('}')[0].split(',')
            current_pose = [float(val) for val in pose_values]
            
            # 获取步进值
            try:
                step = float(self.step_size.get())
            except ValueError:
                self.log("请输入有效的步进值")
                return
            
            # 计算新位置
            new_pose = [
                current_pose[0] + dx * step,
                current_pose[1] + dy * step,
                current_pose[2] + dz * step,
                current_pose[3] + drx * step,
                current_pose[4] + dry * step,
                current_pose[5] + drz * step
            ]
            
            # 移动到新位置
            self.calibrator.robot_move.MovJ(*new_pose)
            self.log(f"��动��位置: {new_pose}")
            
            # 更新位置显示
            for entry, value in zip(self.pos_entries, new_pose):
                entry.delete(0, tk.END)
                entry.insert(0, "{:.3f}".format(value))
            
        except Exception as e:
            self.log(f"移动失败: {str(e)}")

    def set_robot_speed(self):
        """设置机器人运动速度"""
        try:
            speed = int(self.speed_value.get())
            if 1 <= speed <= 100:
                self.calibrator.robot_dashboard.SpeedJ(speed)
                self.calibrator.robot_dashboard.SpeedL(speed)
                self.calibrator.robot_dashboard.AccJ(speed)
                self.calibrator.robot_dashboard.AccL(speed)
                self.log(f"速度设置为: {speed}%")
            else:
                self.log("速度值必须在1-100之间")
        except ValueError:
            self.log("请输入有效的速度值")
        except Exception as e:
            self.log(f"设置速度失败: {str(e)}")
        
    def start_auto_capture(self):
        """开始自动采集数据"""
        try:
            # 保存当前速度设置
            current_speed = self.speed_value.get()
            
            # 设置较低的运动速度(10%)
            self.calibrator.robot_dashboard.SpeedJ(10)
            self.calibrator.robot_dashboard.SpeedL(10)
            self.calibrator.robot_dashboard.AccJ(10)
            self.calibrator.robot_dashboard.AccL(10)
            self.speed_value.delete(0, tk.END)
            self.speed_value.insert(0, "10")
            self.log("已将运动速度设置为10%用于数据采集")
            
            # 定义初始位置
            initial_pose = [-118.141403, -344.319641, 415.981628, -179.962280, 0.054863, 89.871269]
            
            # 先移动到初始位置
            self.log("正在移动到初始位置...")
            self.calibrator.robot_move.MovJ(*initial_pose)
            time.sleep(4)  # 等待到达初始位置
            
            # 获取目标采集点数
            try:
                target_points = int(self.points_count.get())
                if target_points < 4:
                    self.log("采集点数至少需要4个")
                    return
            except ValueError:
                self.log("请输入有效的采集点数")
                return
            
            # 使用初始位置作为中心点
            center_pose = initial_pose
            
            # 设置采集参数
            radius = 50  # 移动半径(mm)
            height_levels = [-30, 0, 30, 60, 90]  # 五个不同的高度层级(mm)
            angle_range = 30  # 姿态角度变化范围(度)
            
            # 开始连续采集
            self.log("开始自动采集...")
            
            def on_capture_complete():
                # 恢复原来的速度设置
                self.calibrator.robot_dashboard.SpeedJ(int(current_speed))
                self.calibrator.robot_dashboard.SpeedL(int(current_speed))
                self.calibrator.robot_dashboard.AccJ(int(current_speed))
                self.calibrator.robot_dashboard.AccL(int(current_speed))
                self.speed_value.delete(0, tk.END)
                self.speed_value.insert(0, current_speed)
                self.log("采集完成，已恢复原速度设置")
            
            self.continuous_capture(center_pose, radius, height_levels, angle_range, target_points, on_complete=on_capture_complete)
            
        except Exception as e:
            self.log(f"自动采集失败: {str(e)}")

    def continuous_capture(self, center_pose, radius, height_levels, angle_range, target_points, on_complete=None):
        """连续采集直到达到目标数量"""
        
        def generate_next_pose(angle, height_level):
            """生成下一个位姿"""
            # 计算XY平面上的位置
            x = center_pose[0] + radius * np.cos(np.radians(angle))
            y = center_pose[1] + radius * np.sin(np.radians(angle))
            z = center_pose[2] + height_level  # 使用指定的高度层级
            
            # 计算姿态角度（添加较小的随机变化）
            rx = center_pose[3] + np.random.uniform(-angle_range/2, angle_range/2)
            ry = center_pose[4] + np.random.uniform(-angle_range/2, angle_range/2)
            rz = center_pose[5] + np.random.uniform(-angle_range/2, angle_range/2)
            
            return [x, y, z, rx, ry, rz]
        
        def execute_continuous_capture():
            # 检查是否已达到目标数量
            current_count = len(self.calibrator.robot_poses)
            if current_count >= target_points:
                self.log(f"已��成目标采集数量: {target_points}组")
                if on_complete:
                    on_complete()
                return
            
            # 如果是新的高度层级的开始，先回到初始位置
            nonlocal current_angle
            if current_angle == 0:
                self.log("移动回初始位置...")
                initial_pose = [-118.141403, -344.319641, 415.981628, -179.962280, 0.054863, 89.871269]
                self.calibrator.robot_move.MovJ(*initial_pose)
                time.sleep(4)  # 增加等待时间到4秒
            
            # 生成下一个位姿
            nonlocal current_height_index
            current_height = height_levels[current_height_index]
            pose = generate_next_pose(current_angle, current_height)
            
            try:
                # 移动到指定位置
                self.calibrator.robot_move.MovJ(*pose)
                
                # 增加等待时间确保位置稳定
                time.sleep(2.5)
                
                # 尝试采集数据
                ret, corners, img = self.calibrator.capture_calibration_data()
                if ret:
                    # 获取当前位姿并更新显示
                    pose_str = self.calibrator.robot_dashboard.GetPose()
                    pose_values = pose_str.split(',{')[1].split('}')[0].split(',')
                    actual_pose = [float(val) for val in pose_values]
                    
                    # 添加到树形列表
                    count = len(self.calibrator.robot_poses)
                    self.data_tree.insert("", "end", values=(
                        count,
                        "{:.3f}".format(actual_pose[0]),
                        "{:.3f}".format(actual_pose[1]),
                        "{:.3f}".format(actual_pose[2]),
                        "{:.3f}".format(actual_pose[3]),
                        "{:.3f}".format(actual_pose[4]),
                        "{:.3f}".format(actual_pose[5])
                    ))
                    
                    self.data_count_label.config(text=f"已采集数据: {count} 组")
                    self.log(f"成功采集第 {count} 组数据 (高度层级: {current_height}mm)")
                else:
                    self.log(f"未检测到标定板 (高度层级: {current_height}mm)")
                
                # 更新下一个位置的角度和高度
                current_angle += 30  # 每30度采集一次
                if current_angle >= 360:
                    current_angle = 0
                    current_height_index = (current_height_index + 1) % len(height_levels)
                    self.log(f"切换到新的高度层级: {height_levels[current_height_index]}mm")
                    
            except Exception as e:
                self.log(f"采集失败: {str(e)}")
            
            finally:
                # 继续下一次采集，增加延时到500ms
                self.root.after(500, execute_continuous_capture)
        
        # 初始化角度和高度索引
        current_angle = 0
        current_height_index = 0
        
        # 开始连续采集
        execute_continuous_capture()

    def save_calibration_data(self):
        """保存标定数据"""
        try:
            from tkinter import filedialog
            import json
            import os
            
            if not self.calibrator.robot_poses:
                self.log("没有数据需要保存")
                return
            
            # 创建保存数据的字典
            save_data = {
                'robot_poses': self.calibrator.robot_poses,
                'image_points': [points.tolist() for points in self.calibrator.image_points],
                'board_size': self.calibrator.board_size,
                'square_size': self.calibrator.square_size
            }
            
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="保存标定数据"
            )
            
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(save_data, f, indent=4, ensure_ascii=False)
                self.log(f"数据已保存到: {file_path}")
                
        except Exception as e:
            self.log(f"保存数��失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def load_calibration_data(self):
        """加载标定数据"""
        try:
            from tkinter import filedialog
            import json
            import numpy as np
            
            # 选择要加载的文件
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json")],
                title="加载标定数据"
            )
            
            if file_path:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 更新标定器参数
                self.calibrator.robot_poses = data['robot_poses']
                self.calibrator.image_points = [np.array(points) for points in data['image_points']]
                self.calibrator.board_size = tuple(data['board_size'])
                self.calibrator.square_size = data['square_size']
                
                # 如果数据中包含标定结果，也加载它们
                if 'calibration_result' in data:
                    self.calibrator.R = np.array(data['calibration_result']['R'])
                    self.calibrator.t = np.array(data['calibration_result']['t'])
                    self.calibrator.calibration_error = data['calibration_result'].get('error', None)
                    self.log("已加载标定结果")
                
                # 更新UI显示
                self.board_width.delete(0, tk.END)
                self.board_width.insert(0, str(self.calibrator.board_size[0]))
                self.board_height.delete(0, tk.END)
                self.board_height.insert(0, str(self.calibrator.board_size[1]))
                self.square_size.delete(0, tk.END)
                self.square_size.insert(0, str(self.calibrator.square_size))
                
                # 更新数据显示
                self.update_data_display()
                
                self.log(f"成功加载数据从: {file_path}")
                self.log(f"加载了 {len(self.calibrator.robot_poses)} 组数据")
                
        except Exception as e:
            self.log(f"加载数据失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def verify_calibration(self):
        """验证手眼标定结果"""
        try:
            # 检查是否有标定结果
            if not hasattr(self.calibrator, 'R') or not hasattr(self.calibrator, 't'):
                self.log("错误：未找到标定结果，请先进行标定")
                return
            
            self.log("\n开始验证标定结果...")
            
            # 1. 移动到初始验证位置
            initial_pose = [-118.141403, -344.319641, 415.981628, -179.962280, 0.054863, 89.871269]
            self.log("移动到初始位置...")
            self.calibrator.robot_move.MovJ(*initial_pose)
            time.sleep(3)
            
            # 2. 设置较低的运动速度
            self.calibrator.robot_dashboard.SpeedJ(10)
            self.calibrator.robot_dashboard.SpeedL(10)
            
            # 3. 进行多点验证
            verification_points = [
                [initial_pose[0], initial_pose[1], initial_pose[2], initial_pose[3], initial_pose[4], initial_pose[5]],  # 初始位置
                [initial_pose[0] + 50, initial_pose[1], initial_pose[2], initial_pose[3], initial_pose[4], initial_pose[5]],  # X+50
                [initial_pose[0], initial_pose[1] + 50, initial_pose[2], initial_pose[3], initial_pose[4], initial_pose[5]],  # Y+50
                [initial_pose[0], initial_pose[1], initial_pose[2] + 50, initial_pose[3], initial_pose[4], initial_pose[5]],  # Z+50
            ]
            
            errors = []
            for i, pose in enumerate(verification_points):
                self.log(f"\n验证点 {i+1}:")
                self.log(f"移动到位置: {pose}")
                
                # 移动到验证位置
                self.calibrator.robot_move.MovJ(*pose)
                time.sleep(2.5)
                
                # 获取实际机器人位姿
                pose_str = self.calibrator.robot_dashboard.GetPose()
                robot_pose = [float(val) for val in pose_str.split(',{')[1].split('}')[0].split(',')]
                
                # 获取相机观察到的标定板位置
                ret, corners, img = self.calibrator.capture_calibration_data()
                if ret:
                    # 计算标定板位姿
                    ret, rvec, tvec = cv2.solvePnP(
                        self.calibrator.obj_points,
                        corners,
                        self.calibrator.camera_matrix,
                        self.calibrator.dist_coeffs
                    )
                    
                    if ret:
                        # 将旋转向量转换为旋转矩阵
                        R_marker, _ = cv2.Rodrigues(rvec)
                        
                        # 使用标定结果计算预测位置
                        predicted_R = self.calibrator.R @ R_marker
                        predicted_t = self.calibrator.R @ tvec + self.calibrator.t
                        
                        # 计算误差
                        R_error = np.linalg.norm(predicted_R - R_marker)
                        t_error = np.linalg.norm(predicted_t - tvec)
                        
                        errors.append((R_error, t_error))
                        
                        self.log(f"旋转误差: {R_error:.6f} rad")
                        self.log(f"平移误差: {t_error:.6f} m")
                    else:
                        self.log("无法计算标定板位姿")
                else:
                    self.log("未检测到标定板")
            
            # 4. 计算平均误差
            if errors:
                avg_R_error = np.mean([e[0] for e in errors])
                avg_t_error = np.mean([e[1] for e in errors])
                
                self.log("\n验证结果总结:")
                self.log(f"平均旋转误差: {avg_R_error:.6f} rad")
                self.log(f"平均平移误差: {avg_t_error:.6f} m")
                
                # 5. 给出评估结果
                if avg_R_error < 0.1 and avg_t_error < 0.01:
                    self.log("\n标定结果评估: 优秀")
                elif avg_R_error < 0.2 and avg_t_error < 0.02:
                    self.log("\n标定结果评估: 良好")
                else:
                    self.log("\n标定结果评估: 需要重新标定")
            
            # 恢复速度设置
            speed = self.speed_value.get()
            self.calibrator.robot_dashboard.SpeedJ(int(speed))
            self.calibrator.robot_dashboard.SpeedL(int(speed))
            
        except Exception as e:
            self.log(f"验证过程出错: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

if __name__ == "__main__":
    app = CalibrationGUI()
    app.run()