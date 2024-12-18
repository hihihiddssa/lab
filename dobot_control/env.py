import time
from typing import Any, Dict, Optional

import numpy as np

from dobot_control.cameras.camera import CameraDriver
from dobot_control.robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class RobotEnv:
    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        return 0

    def step(self, joints: np.ndarray, flag_in: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        flag_in 参数在这段代码中没有详细解释，但根据上下文和常见的机器人控制实践，我可以推测它的可能用途：
                控制模式标志：
                可能用于指示机器人应该以哪种模式运行，例如位置控制、速度控制或力控制。
                功能启用/禁用：
                可能用于启用或禁用某些特定的机器人功能，如夹持器、安全系统等。
                运动类型指示：
                可能用于指示当前命令是否为特殊类型的运动，如直线运动、圆弧运动等。
        Args:
            joints: joint angles command to step the environment with.
            flag_in

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        tic = time.time()
        self._robot.command_joint_state(joints, flag_in)
        toc = time.time()

        # print("command_joint_state", toc-tic)
        return self.get_obs()


    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        observations = {}
        # for name, camera in self._camera_dict.items():
        #     image, depth = camera.read()
        #     observations[f"{name}_rgb"] = image
            # observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        assert "joint_positions" in robot_obs
        assert "joint_velocities" in robot_obs
        assert "ee_pos_quat" in robot_obs
        observations["joint_positions"] = robot_obs["joint_positions"]
        observations["joint_velocities"] = robot_obs["joint_velocities"]
        observations["ee_pos_quat"] = robot_obs["ee_pos_quat"]
        observations["gripper_position"] = robot_obs["gripper_position"]
        return observations

    def set_do_status(self, which_do):
        self._robot.set_do_status(np.array(which_do))

    def get_XYZrxryrz_state(self) -> Dict[str, Any]:
        """Get the current X Y Z rx ry rz state of the robot.
        Returns:
            T: The current X Y Z rx ry rz state of the  robot.
        """
        robot_pos = self._robot.get_XYZrxryrz_state()
        return robot_pos

def main() -> None:
    pass


if __name__ == "__main__":
    main()
