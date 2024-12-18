import numpy as np
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.robots.robot_node import ZMQClientRobot
from dobot_control.env import RobotEnv
from scripts.manipulate_utils import robot_pose_init
import time


# 定义初始安全位置的关节角度（已从角度转换为弧度）（degree to rad）
reset_joints_left = np.deg2rad([-90, 0, -90, 0, 90, 90, 57])
reset_joints_right = np.deg2rad([90, 0, 90, 0, -90, -90, 57])


def move_to_reset_position():
    # 创建机器人客户端和环境
    robot_client = ZMQClientRobot(port=6001, host="127.0.0.1")
    env = RobotEnv(robot_client)
    # 初始化机器人位姿
    robot_pose_init(env)
    # 创建左右臂的机器人代理
    left_agent = DobotAgent(which_hand="LEFT")
    right_agent = DobotAgent(which_hand="RIGHT")

    protect_err = False
    if protect_err:
        env.set_do_status([3, 0])
        env.set_do_status([2, 0])
        env.set_do_status([1, 1])
        time.sleep(1)
        exit()
    # 移动左臂到初始安全位置
    left_agent.move_joints(reset_joints_left)
    # 移动右臂到初始安全位置
    right_agent.move_joints(reset_joints_right)



    # 等待机器人运动完成（这里假设机器人运动到目标位置需要一定时间，可以根据实际情况调整等待时间）
    time.sleep(5)


if __name__ == "__main__":
    move_to_reset_position()

