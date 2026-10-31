"""SO-100 Xbox teleoperation. Run --help for setup and hardware-free checks."""

import argparse
import math
import time
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def deadzone(value, threshold=0.15):
    return 0.0 if abs(value) <= threshold else math.copysign(
        (abs(value) - threshold) / (1 - threshold), value
    )


def compute_action(observation, kinematics, inputs, dt):
    """Build a bounded action from fresh feedback; inputs are normalized velocities.

    Cartesian targets start at measured FK each tick, preventing unreachable targets
    from accumulating. Arm positions use degrees; the gripper uses percent open.
    """
    current = np.array([observation[f"{name}.pos"] for name in JOINTS], dtype=float)
    gripper = float(observation["gripper.pos"])
    if not np.isfinite(np.append(current, gripper)).all():
        raise ValueError("Robot observation contains non-finite positions")
    target = current.copy()
    if inputs.get("enabled", False):
        pose = np.array(kinematics.forward_kinematics(current), dtype=float, copy=True)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Forward kinematics must return a finite 4x4 transform")
        pose[:3, 3] += np.array([inputs.get(k, 0.0) for k in ("x", "y", "z")]) * 0.05 * dt
        if any(inputs.get(k, 0.0) for k in ("x", "y", "z")):
            solution = np.asarray(kinematics.inverse_kinematics(current, pose), dtype=float)
            if solution.shape != (5,) or not np.isfinite(solution).all():
                raise ValueError("Inverse kinematics returned invalid joint positions")
            target = solution.copy()
        # Bumpers directly control wrist_roll in degrees, rather than a matrix row.
        target[4] += inputs.get("roll", 0.0) * 30.0 * dt
        target = current + np.clip(target - current, -30.0 * dt, 30.0 * dt)
        gripper += inputs.get("grip", 0.0) * 30.0 * dt
    action = {f"{name}.pos": float(value) for name, value in zip(JOINTS, target, strict=True)}
    action["gripper.pos"] = float(np.clip(gripper, 0, 100))
    return action


class SimulatedArm:
    """Deterministic loopback backend, not a physical SO-100 dynamics model."""

    def __init__(self):
        self.observation = {f"{name}.pos": 0.0 for name in JOINTS}
        self.observation["gripper.pos"] = 50.0

    def get_observation(self):
        return self.observation.copy()

    def send_action(self, action):
        self.observation.update(action)

    def forward_kinematics(self, joints):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(joints[:3]) / 1000
        return pose

    def inverse_kinematics(self, current_joint_pos, desired_ee_pose):
        result = current_joint_pos.copy()
        result[:3] = desired_ee_pose[:3, 3] * 1000
        return result


def read_controller(controller):
    """SDL game-controller mapping (consistent across supported Xbox devices)."""
    import pygame as sdl

    axis = lambda index: float(np.clip(controller.get_axis(index) / 32767, -1, 1))
    return {
        "enabled": bool(controller.get_button(sdl.CONTROLLER_BUTTON_X)),
        "x": deadzone(axis(sdl.CONTROLLER_AXIS_LEFTX)),
        "y": -deadzone(axis(sdl.CONTROLLER_AXIS_LEFTY)),
        "z": deadzone(max(0, axis(sdl.CONTROLLER_AXIS_TRIGGERRIGHT)))
        - deadzone(max(0, axis(sdl.CONTROLLER_AXIS_TRIGGERLEFT))),
        "roll": int(controller.get_button(sdl.CONTROLLER_BUTTON_RIGHTSHOULDER))
        - int(controller.get_button(sdl.CONTROLLER_BUTTON_LEFTSHOULDER)),
        "grip": int(controller.get_button(sdl.CONTROLLER_BUTTON_B))
        - int(controller.get_button(sdl.CONTROLLER_BUTTON_A)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Robot serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--urdf", type=Path, help="SO-100 URDF matching the calibrated arm")
    parser.add_argument("--frame", default="gripper_frame_link", help="End-effector URDF frame")
    parser.add_argument("--robot-id", default="my_arm", help="Existing LeRobot calibration ID")
    parser.add_argument("--dry-run", action="store_true", help="Simulated arm; never connects to hardware")
    parser.add_argument("--demo", action="store_true", help="Finite scripted inputs; requires --dry-run")
    parser.add_argument("--steps", type=int, default=120, help="Number of demo ticks")
    args = parser.parse_args(argv)
    if args.demo and not args.dry_run:
        parser.error("--demo requires --dry-run")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not args.dry_run and (not args.port or not args.urdf or not args.urdf.is_file()):
        parser.error("hardware mode requires --port and --urdf pointing to an existing file")

    robot = controller = pygame = None
    try:
        if not args.demo:
            import pygame
            from pygame._sdl2 import controller as sdl

            pygame.init()
            sdl.init()
            index = next((i for i in range(sdl.get_count()) if sdl.is_controller(i)), None)
            if index is None:
                raise RuntimeError("No supported SDL controller found")
            controller = sdl.Controller(index)
            print("Hold X to move. Left stick: XY; triggers: Z; bumpers: wrist; A/B: close/open.")
            print("Start or Ctrl+C exits. Release X to hold position.")
        if args.dry_run:
            robot = kinematics = SimulatedArm()
        else:
            from lerobot.model.kinematics import RobotKinematics
            from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

            kinematics = RobotKinematics(str(args.urdf), args.frame, list(JOINTS))
            robot = SO100Follower(SO100FollowerConfig(
                port=args.port, id=args.robot_id, use_degrees=True,
                max_relative_target=2.0, disable_torque_on_disconnect=True,
            ))
            # Calibration should be performed explicitly, not inside a teleoperation session.
            robot.connect(calibrate=False)
            if not robot.is_calibrated:
                raise RuntimeError("Calibrate the arm with LeRobot before teleoperation")
        previous = time.monotonic()
        tick = 0
        while True:
            started = time.monotonic()
            dt = min(max(started - previous, 0.0), 1 / 30)
            previous = started
            if args.demo:
                dt = 1 / 30
                inputs = {"enabled": True, "x": 0.5, "z": 0.25, "roll": 0.2, "grip": 0.5}
            else:
                events = pygame.event.get()
                if any(e.type in (pygame.QUIT, pygame.CONTROLLERDEVICEREMOVED) for e in events):
                    break
                if not controller.attached():
                    raise RuntimeError("Controller disconnected")
                if controller.get_button(pygame.CONTROLLER_BUTTON_START):
                    break
                inputs = read_controller(controller)
            robot.send_action(compute_action(robot.get_observation(), kinematics, inputs, dt))
            tick += 1
            if args.demo and tick >= args.steps:
                print(f"Simulated {tick} ticks: {robot.get_observation()}")
                break
            time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        try:
            if robot is not None and not args.dry_run and robot.is_connected:
                robot.disconnect()
        finally:
            if controller is not None:
                controller.quit()
            if pygame is not None:
                pygame.quit()


if __name__ == "__main__":
    main()
