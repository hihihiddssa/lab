import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import cv2
import time
from dataclasses import dataclass
import numpy as np
import tyro
import threading
from dobot_control.agents.agent import BimanualAgent
from scripts.format_obs import save_frame
from dobot_control.env import RobotEnv
from dobot_control.robots.robot_node import ZMQClientRobot
from scripts.function_util import mismatch_data_write, wait_period, log_write, mk_dir
from scripts.manipulate_utils import robot_pose_init, pose_check, dynamic_approach, obs_action_check, servo_action_check, load_ini_data_hands, set_light, load_ini_data_camera
from dobot_control.agents.dobot_agent import DobotAgent
from dobot_control.cameras.realsense_camera import RealSenseCamera
import datetime
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Args:





    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    show_img: bool = False
    save_data_path = str(Path(__file__).parent.parent.parent)+"/datasets/"
    project_name = "0926_orange_plasticbag_L"


# Thread button: [lock or nor, servo or not, record or not]
# 0: lock, 1: unlock
# 0: stop servo, 1: servo
# 0: stop recording, 1: recording
what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
dt_time = np.array([20240507161455])

def button_monitor_realtime(agent):
    # servo
    last_keys_status = np.array(([0, 0], [0, 0]))
    start_press_status = np.array(([0, 0], [0, 0]))  # start press
    keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))

    while 1:
        # time.sleep(0.010)
        now_keys = agent.get_keys()
        dev_keys = now_keys - last_keys_status
        # button a
        for i in range(2):
            if dev_keys[i, 0] == -1:  # button a: start
                tic = time.time()
                start_press_status[i, 0] = 1
            if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # button a: end
                start_press_status[i, 0] = 0
                toc = time.time()
                if toc-tic < 0.5:
                    keys_press_count[i, 0] += 1
                    # print(i, keys_press_count[i, 0], "short press", toc-tic)
                    if keys_press_count[i, 0] % 2 == 1:
                        what_to_do[i, 0] = 1
                        # log_write(__file__, "ButtonA: ["+str(i)+"] unlock")
                        print("ButtonA: [" + str(i) + "] unlock", what_to_do)
                    else:
                        what_to_do[i, 0] = 0
                        # log_write(__file__, "ButtonA: [" + str(i) + "] lock")
                        print("ButtonA: [" + str(i) + "] lock", what_to_do)

                elif toc-tic > 1:
                    keys_press_count[i, 1] += 1
                    # print(i, keys_press_count[i, 1], "long press", toc-tic)
                    if keys_press_count[i, 1] % 2 == 1:
                        what_to_do[i, 1] = 1
                        # log_write(__file__, "ButtonA: [" + str(i) + "] servo")
                        print("ButtonA: [" + str(i) + "] servo")
                    else:
                        what_to_do[i, 1] = 0
                        # log_write(__file__, "ButtonA: [" + str(i) + "] stop servo")
                        print("ButtonA: [" + str(i) + "] stop servo")

        # button B
        # more than one start servo
        for i in range(2):
            if dev_keys[i, 1] == -1:  # B button pressed
                start_press_status[i, 1] = 1
            if dev_keys[i, 1] == 1:
                start_press_status[i, 1] = 0
                if keys_press_count[0, 2] % 2 == 1:
                    if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                        what_to_do[0, 2] = 1
                        # log_write(__file__, "ButtonB: [" + str(i) + "] recording")
                        # new recording
                        now_time = datetime.datetime.now()
                        dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                        keys_press_count[0, 2] += 1
                else:
                    what_to_do[0, 2] = 0
                    keys_press_count[0, 2] += 1
                    # log_write(__file__, "ButtonB: [" + str(i) + "] stop recording")

        last_keys_status = now_keys


# Thread: camera
npy_list = np.array([np.zeros(480*640*3), np.zeros(480*640*3), np.zeros(480*640*3), np.zeros(480*640*3)])
npy_len_list = np.array([0, 0, 0,0])
img_list = np.array([np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3)), np.zeros((480, 640, 3))])


def run_thread_cam(rs_cam, which_cam):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    while 1:
        image_cam, _ = rs_cam.read()
        image_cam = image_cam[:, :, ::-1]
        img_list[which_cam] = image_cam
        _, image_ = cv2.imencode('.jpg', image_cam, encode_param)
        npy_list[which_cam][:len(image_)] = image_
        npy_len_list[which_cam] = len(image_)


def main(args):
    # create dataset file path
    save_dir = args.save_data_path+args.project_name+"/collect_data"
    mk_dir(save_dir)

    # camera init
    camera_dict = load_ini_data_camera()
    rs1 = RealSenseCamera(flip=True, device_id=camera_dict["top"])
    rs2 = RealSenseCamera(flip=False, device_id=camera_dict["left"])
    rs3 = RealSenseCamera(flip=True, device_id=camera_dict["right"])
    rs4= RealSenseCamera(flip=False, device_id=camera_dict["bottom"])


    thread_cam_top = threading.Thread(target=run_thread_cam, args=(rs1, 0))
    thread_cam_left = threading.Thread(target=run_thread_cam, args=(rs2, 1))
    thread_cam_right = threading.Thread(target=run_thread_cam, args=(rs3, 2))
    thread_cam_bottom = threading.Thread(target=run_thread_cam, args=(rs4, 3))
    thread_cam_top.start()
    thread_cam_left.start()
    thread_cam_right.start()
    thread_cam_bottom.start()
    show_canvas = np.zeros((480, 640*4, 3), dtype=np.uint8)
    time.sleep(2)
    print("camera thread init success...")

    while 1:

        args.show_img=1
        # img show
        if args.show_img:
            show_canvas[:, :640] = np.asarray(img_list[0], dtype="uint8")
            show_canvas[:, 640:640 * 2] = np.asarray(img_list[1], dtype="uint8")
            show_canvas[:, 640 * 2:640 * 3] = np.asarray(img_list[2], dtype="uint8")
            show_canvas[:, 640 * 3:] = np.asarray(img_list[3], dtype="uint8")
            cv2.imshow("0", show_canvas)
            cv2.waitKey(1)


        toc = time.time()
        # total_time = toc-tic
        # print("total time: ", total_time)


if __name__ == "__main__":
    main(tyro.cli(Args))
