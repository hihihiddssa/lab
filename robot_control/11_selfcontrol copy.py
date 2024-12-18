import os
import sys


from dobotapi.robot_control.dobot_api import DobotApiDashboard, DobotApiMove
import time
import tkinter as tk
from tkinter import ttk
import threading
import json
import os

class DobotControl:
    def __init__(self):
        # 机器人配置
        self.ip = "192.168.5.1"
        self.dashboard_port = 29999
        self.move_port = 30003
        
        # 配置文件路径
        self.config_file = "robot_initial_positions.json"
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Dobot 控制界面")
        
        # 创建API实例
        self.dashboard = DobotApiDashboard(self.ip, self.dashboard_port)
        self.move = DobotApiMove(self.ip, self.move_port)
        
        # 添加节流控制变量
        self.move_timer = None
        self.MOVE_DELAY = 100  # 100ms 的延迟
        
        # 添加连续调节控制变量
        self.adjust_timer = None
        self.ADJUST_INTERVAL = 50  # 50ms 的调节间隔
        
        # 加载初始位置
        self.initial_positions = self.load_initial_positions()
        
        # 创建特殊样式的按钮
        style = ttk.Style()
        style.configure('Important.TButton', 
                       background='#4CAF50',
                       foreground='white',
                       padding=5)
        
        self.setup_ui()
        self.initialize_robot()
        
    def setup_ui(self):
        # 创建初始位置设置框架
        init_frame = ttk.LabelFrame(self.root, text="初始位置设置")
        init_frame.pack(padx=20, pady=10, fill="x")
        
        self.init_entries = []
        entry_frame = ttk.Frame(init_frame)
        entry_frame.pack(padx=5, pady=5)
        
        # 创建6个输入框用于设置初始位置
        for i in range(6):
            ttk.Label(entry_frame, text=f"J{i+1}:").grid(row=0, column=i*2, padx=2)
            entry = ttk.Entry(entry_frame, width=8)
            entry.grid(row=0, column=i*2+1, padx=2)
            entry.insert(0, str(self.initial_positions[i]))
            self.init_entries.append(entry)
        
        # 添加设置和获取按钮框架
        init_button_frame = ttk.Frame(init_frame)
        init_button_frame.pack(pady=5)
        
        # 设置初始位置按钮
        ttk.Button(init_button_frame, text="设置初始位置", 
                  command=self.set_initial_position).pack(side="left", padx=5)
        
        # 获取当前位置按钮
        ttk.Button(init_button_frame, text="获取当前位置作为初始位置", 
                  command=self.get_current_as_initial).pack(side="left", padx=5)
        
        # 返回初始位置按钮
        ttk.Button(init_button_frame, text="返回初始位置", 
                  command=self.return_to_initial,
                  style='Important.TButton').pack(side="left", padx=5)
        
        # 创建位置控制框架
        position_frame = ttk.LabelFrame(self.root, text="位置控制")
        position_frame.pack(padx=20, pady=10, fill="x")
        
        # 创建输入框架
        input_frame = ttk.Frame(position_frame)
        input_frame.pack(padx=5, pady=5)
        
        # 创建位置输入框
        self.position_entries = []
        position_labels = ['X:', 'Y:', 'Z:', 'Rx:', 'Ry:', 'Rz:']
        
        for i, label in enumerate(position_labels):
            ttk.Label(input_frame, text=label).grid(row=0, column=i*2, padx=2)
            entry = ttk.Entry(input_frame, width=8)
            entry.grid(row=0, column=i*2+1, padx=2)
            self.position_entries.append(entry)
        
        # 创建按钮框架
        pos_button_frame = ttk.Frame(position_frame)
        pos_button_frame.pack(pady=5)
        
        # 添加运动控制按钮
        ttk.Button(pos_button_frame, text="移动到指定位置", 
                  command=self.move_to_position).pack(side="left", padx=5)
        
        ttk.Button(pos_button_frame, text="获取当前位置", 
                  command=self.get_current_to_entries).pack(side="left", padx=5)
        
        # 创建关节控制框架
        joint_frame = ttk.LabelFrame(self.root, text="关节控制")
        joint_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.joint_vars = []
        self.joint_labels = []
        
        # 创建6个关节的滑块控制
        for i in range(6):
            # 为每个关节创建子框架
            joint_subframe = ttk.Frame(joint_frame)
            joint_subframe.pack(padx=10, pady=5, fill="x")
            
            joint_var = tk.DoubleVar()
            self.joint_vars.append(joint_var)
            
            # 减少按钮
            minus_btn = ttk.Button(joint_subframe, text="-", width=3)
            minus_btn.bind('<Button-1>', lambda e, i=i: self.start_continuous_adjust(i, -1))
            minus_btn.bind('<ButtonRelease-1>', self.stop_continuous_adjust)
            minus_btn.pack(side="left", padx=2)
            
            # 标签显示当前值
            label = ttk.Label(joint_subframe, text=f"J{i+1}: 0.0°", width=12)
            label.pack(side="left", padx=5)
            self.joint_labels.append(label)
            
            # 滑块
            slider = ttk.Scale(joint_subframe, from_=-180, to=180, 
                             variable=joint_var, orient="horizontal",
                             command=lambda v, i=i: self.on_slider_change(i),
                             length=400)
            slider.pack(side="left", padx=5, fill="x", expand=True)
            
            # 增加按钮
            plus_btn = ttk.Button(joint_subframe, text="+", width=3)
            plus_btn.bind('<Button-1>', lambda e, i=i: self.start_continuous_adjust(i, 1))
            plus_btn.bind('<ButtonRelease-1>', self.stop_continuous_adjust)
            plus_btn.pack(side="left", padx=2)
        
        # 创建按钮框架
        button_frame = ttk.Frame(self.root)
        button_frame.pack(padx=20, pady=10, fill="x")
        
        # 获取位置信息按钮
        ttk.Button(button_frame, text="获取当前位置", 
                  command=self.get_current_position,
                  width=20).pack(side="left", padx=5)
        
        # 位置信息显示框
        self.position_text = tk.Text(self.root, height=5, width=60)
        self.position_text.pack(padx=20, pady=10, fill="both", expand=True)
        
        # 创建夹爪控制框架
        gripper_frame = ttk.LabelFrame(self.root, text="夹爪控制")
        gripper_frame.pack(padx=20, pady=10, fill="x")
        
        # 添加夹爪开合按钮
        ttk.Button(gripper_frame, text="打开夹爪", 
                  command=lambda: self.control_gripper(1)).pack(side="left", padx=5)
        ttk.Button(gripper_frame, text="关闭夹爪", 
                  command=lambda: self.control_gripper(0)).pack(side="left", padx=5)
        
        # 设置窗口最小大小
        self.root.minsize(600, 500)
    
    def initialize_robot(self):
        try:
            # 清除错误
            self.dashboard.ClearError()
            
            # 使能机器人
            self.dashboard.EnableRobot()
            time.sleep(1)
            
            # 设置速度和加速度
            self.dashboard.SpeedFactor(25)
            self.dashboard.AccL(25)
            
            # 设置为TCP模式
            self.dashboard.User(0)
            self.dashboard.Tool(0)
            
            # 移动到初始位置
            for i, pos in enumerate(self.initial_positions):
                self.joint_vars[i].set(pos)
            self.delayed_move()
            
            # 获取并显示初始位置
            self.get_current_position()
            
        except Exception as e:
            print(f"初始化错误: {str(e)}")
    
    def on_slider_change(self, joint_index):
        # 更新标签显示
        value = self.joint_vars[joint_index].get()
        self.joint_labels[joint_index].config(
            text=f"J{joint_index+1}: {value:.1f}°")
        
        # 取消之前的定时器
        if self.move_timer is not None:
            self.root.after_cancel(self.move_timer)
        
        # 设置新的定时器
        self.move_timer = self.root.after(self.MOVE_DELAY, self.delayed_move)
    
    def delayed_move(self):
        # 获取所有关节值
        joint_values = [var.get() for var in self.joint_vars]
        
        # 使用线程执行机器人移动，避免界面卡顿
        threading.Thread(target=self.move_robot, 
                       args=(joint_values,), 
                       daemon=True).start()
        
        # 重置定时器
        self.move_timer = None
    
    def move_robot(self, joint_values):
        try:
            # 执行关节运动
            self.move.JointMovJ(*joint_values)
        except Exception as e:
            print(f"移动错误: {str(e)}")
    
    def get_current_position(self):
        try:
            # 获取当前位置信息
            response = self.dashboard.GetPose()
            # 解析返回的字符串，格式为: "0,{x,y,z,rx,ry,rz},GetPose();"
            values = response.split(',')
            # 提取花括号中的数值
            pose_str = values[1].strip('{}')
            pose = [float(x) for x in pose_str.split(',')]
            
            self.position_text.delete(1.0, tk.END)
            self.position_text.insert(tk.END, f"当前位置:\n")
            self.position_text.insert(tk.END, f"X: {pose[0]:.3f}, Y: {pose[1]:.3f}, Z: {pose[2]:.3f}\n")
            self.position_text.insert(tk.END, f"Rx: {pose[3]:.3f}, Ry: {pose[4]:.3f}, Rz: {pose[5]:.3f}")
        except Exception as e:
            print(f"获取位置错误: {str(e)}")
    
    def run(self):
        self.root.mainloop()
    
    def __del__(self):
        # 关闭连接
        try:
            self.dashboard.DisableRobot()
            self.dashboard.close()
            self.move.close()
        except:
            pass
    
    def start_continuous_adjust(self, joint_index, direction):
        """
        开始连续调节
        """
        def adjust():
            if self.adjust_timer is not None:  # 确认还在调节状态
                self.adjust_joint(joint_index, direction)
                self.adjust_timer = self.root.after(self.ADJUST_INTERVAL, adjust)
        
        self.stop_continuous_adjust(None)  # 停止之前的调节
        self.adjust_joint(joint_index, direction)  # 立即执行一次调节
        self.adjust_timer = self.root.after(self.ADJUST_INTERVAL, adjust)  # 启动连续调节

    def stop_continuous_adjust(self, event):
        """
        停止连续调节
        """
        if self.adjust_timer is not None:
            self.root.after_cancel(self.adjust_timer)
            self.adjust_timer = None

    def adjust_joint(self, joint_index, direction):
        """
        微调关节角度
        joint_index: 关节索引
        direction: 1 表示增加，-1 表示减少
        """
        current_value = self.joint_vars[joint_index].get()
        # 每次调整1度
        new_value = current_value + (direction * 1.0)
        # 确保在有效范围内
        new_value = max(-180, min(180, new_value))
        self.joint_vars[joint_index].set(new_value)
        # 触发滑块变化事件
        self.on_slider_change(joint_index)

    def load_initial_positions(self):
        """
        从配置文件加载初始位置
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        except Exception as e:
            print(f"加载初始位置时出错: {str(e)}")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def save_initial_positions(self):
        """
        保存初始位置到配置文件
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.initial_positions, f)
            print("初始位置已保存到配置文件")
        except Exception as e:
            print(f"保存初始位置时出错: {str(e)}")

    def set_initial_position(self):
        """
        设置并移动到初始位置
        """
        try:
            # 获取输入框中的值
            for i, entry in enumerate(self.init_entries):
                value = float(entry.get())
                if -180 <= value <= 180:
                    self.initial_positions[i] = value
                    self.joint_vars[i].set(value)
                else:
                    print(f"关节 {i+1} 的值必须在 -180 到 180 之间")
                    return
            
            # 移动到初始位置
            self.delayed_move()
            print("已设置初始位置:", self.initial_positions)
            
            # 保存到配置文件
            self.save_initial_positions()
            
        except ValueError as e:
            print(f"设置初始位置时出错: {str(e)}")

    def get_current_as_initial(self):
        """
        获取当前位置作为初始位置
        """
        try:
            # 获取当前关节值
            current_values = [var.get() for var in self.joint_vars]
            
            # 更新输入框和初始位置值
            for i, value in enumerate(current_values):
                self.initial_positions[i] = value
                self.init_entries[i].delete(0, tk.END)
                self.init_entries[i].insert(0, f"{value:.1f}")
            
            print("已更新初始位置:", self.initial_positions)
            
            # 保存到配置文件
            self.save_initial_positions()
            
        except Exception as e:
            print(f"获取当前位置时出错: {str(e)}")

    def move_to_position(self):
        """
        移动到指定的笛卡尔位置
        """
        try:
            # 获取输入框中的值
            position_values = []
            for entry in self.position_entries:
                value = float(entry.get())
                position_values.append(value)
            
            # 使用线程执行机器人移动
            threading.Thread(target=self.move_robot_position, 
                           args=(position_values,), 
                           daemon=True).start()
            
            print("正在移动到位置:", position_values)
            
        except ValueError as e:
            print(f"输入值错误: {str(e)}")

    def move_robot_position(self, position_values):
        """
        执行笛卡尔空间运动
        """
        try:
            # 使用MovL命令执行直线运动
            self.move.MovL(*position_values)
        except Exception as e:
            print(f"移动错误: {str(e)}")

    def get_current_to_entries(self):
        """
        获取当前位置并填入输入框
        """
        try:
            # 获取当前位置信息
            pose = self.dashboard.GetPose()
            
            # 更新输入框
            for i, value in enumerate(pose):
                self.position_entries[i].delete(0, tk.END)
                self.position_entries[i].insert(0, f"{value:.3f}")
            
            print("已获取当前位置")
            
        except Exception as e:
            print(f"获取位置错误: {str(e)}")

    def return_to_initial(self):
        """
        返回到初始位置
        """
        try:
            # 设置所有关节值为初始位置
            for i, pos in enumerate(self.initial_positions):
                self.joint_vars[i].set(pos)
            
            # 更新显示
            for i, pos in enumerate(self.initial_positions):
                self.joint_labels[i].config(text=f"J{i+1}: {pos:.1f}°")
            
            # 移动机器人
            self.delayed_move()
            print("正在返回初始位置:", self.initial_positions)
            
        except Exception as e:
            print(f"返回初始位置时出错: {str(e)}")

    def control_gripper(self, state):
        """
        控制夹爪的开合
        state: 1 表示打开夹爪，0 表示关闭夹爪
        """
        try:
            # 使用 DO 输出控制夹爪
            # 假设 DO1 用于控制夹爪
            self.dashboard.SetDO(1, state)
            print(f"夹爪已 {'打开' if state == 1 else '关闭'}")
        except Exception as e:
            print(f"夹爪控制错误: {str(e)}")

if __name__ == "__main__":
    app = DobotControl()
    app.run()