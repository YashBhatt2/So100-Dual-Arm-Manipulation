import pygame
import time
import numpy as np

# LeRobot v0.4+ API
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.model.kinematics import RobotKinematics

# --- Configuration ---
PORT = "/dev/ttyACM0"
# NOTE: Update this path to the actual location of the SO-100 URDF file
URDF_PATH = "/home/tech/lerobot/SO100/so100.urdf" 

SPEED_XY = 0.005       # Meters to move in X/Y per frame
SPEED_Z = 0.005        # Meters to move in Z per frame
SPEED_ROLL = 2.0       # Degrees to roll wrist per frame
SPEED_GRIPPER = 2.0    # Degrees to move gripper per frame
DEADZONE = 0.15        # Ignore minor stick/trigger drift
WINDUP_LIMIT = 10.0    # Max allowed joint drift (degrees)

def main():
    # 1. Initialize Controller (Xbox mapping)
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("No controller detected. Please plug in or pair your Xbox controller.")
        return
        
    controller = pygame.joystick.Joystick(0)
    controller.init()
    print(f"Connected to: {controller.get_name()}")

    # 2. Initialize Robot
    print(f"Connecting to SO-100 on {PORT}...")
    robot_cfg = SO100FollowerConfig(port=PORT, id="my_arm")
    robot = SO100Follower(robot_cfg)
    robot.connect()
    
    # 3. Initialize Kinematics Solver
    print(f"Loading kinematics solver from {URDF_PATH}...")
    kin_solver = RobotKinematics(urdf_path=URDF_PATH, target_frame_name="jaw", joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
    kin_solver = RobotKinematics(
            urdf_path=URDF_PATH,
            target_frame_name="jaw", joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], # e.g., "wrist_roll"
    )     
    # Read current joint positions to seed Forward Kinematics 
    obs = robot.get_observation()
    current_joints = np.array([
        obs.get("shoulder_pan.pos", 0.0),
        obs.get("shoulder_lift.pos", 0.0),
        obs.get("elbow_flex.pos", 0.0),
        obs.get("wrist_flex.pos", 0.0),
        obs.get("wrist_roll.pos", 0.0)
    ], dtype=np.float64)
    
    current_gripper = obs.get("gripper.pos", 0.0)
    target_gripper = current_gripper
    
    # Get initial Cartesian pose: [x, y, z, roll, pitch, yaw]
    target_pose = kin_solver.forward_kinematics(current_joints)
    target_joints = current_joints.copy()
    
    print("Starting IK Teleoperation. Press CTRL+C to quit.")
    
    try:
        while True:
            pygame.event.pump()
            
            # --- 4. Read Xbox Inputs ---
            # Joystick X-Y (Left Stick)
            left_x = controller.get_axis(0)  
            left_y = controller.get_axis(1)  
            
            # Triggers Z+ and Z- (Axes 4 & 5 rest at -1.0 in Pygame SDL2 standard)
            lt_val = controller.get_axis(4)
            rt_val = controller.get_axis(5)
            # Normalize triggers to a 0.0 to 1.0 range
            z_minus = (lt_val + 1) / 2.0 
            z_plus = (rt_val + 1) / 2.0 
            
            # Bumpers: Wrist Roll
            lb = controller.get_button(4) 
            rb = controller.get_button(5) 
            
            # Gamepad Buttons: End Effector (A=0, B=1)
            a_button = controller.get_button(0)
            b_button = controller.get_button(1)

            # --- 5. Update Target Cartesian Pose ---
            # X-Y Translation
            if abs(left_x) > DEADZONE:  
                target_pose[0] += left_x * SPEED_XY
            if abs(left_y) > DEADZONE:  
                target_pose[1] -= left_y * SPEED_XY 
                
            # Z+ / Z- Translation
            if z_plus > DEADZONE:
                target_pose[2] += z_plus * SPEED_Z
            if z_minus > DEADZONE:
                target_pose[2] -= z_minus * SPEED_Z
                
            # Wrist Roll (+ / - on Pose index 3)
            if rb: target_pose[3] += SPEED_ROLL
            if lb: target_pose[3] -= SPEED_ROLL
                
            # End Effector (Gripper)
            if a_button: target_gripper -= SPEED_GRIPPER
            if b_button: target_gripper += SPEED_GRIPPER

            # --- 6. Compute Inverse Kinematics ---
            # Converts the desired 6D Cartesian pose back into safe joint angles
            ik_solution = kin_solver.inverse_kinematics(target_pose, initial_joints=current_joints)
           
            if ik_solution is not None:
                target_joints = ik_solution
            
            # --- 7. Prevent Integral Windup (Safety Logic) ---
            obs = robot.get_observation()
            actual_joints = [
                obs.get("shoulder_pan.pos", target_joints[0]),
                obs.get("shoulder_lift.pos", target_joints[1]),
                obs.get("elbow_flex.pos", target_joints[2]),
                obs.get("wrist_flex.pos", target_joints[3]),
                obs.get("wrist_roll.pos", target_joints[4])
            ]
            
            for i in range(5):
                diff = target_joints[i] - actual_joints[i]
                if abs(diff) > WINDUP_LIMIT:
                    target_joints[i] = actual_joints[i] + (WINDUP_LIMIT if diff > 0 else -WINDUP_LIMIT)

            # --- 8. Send Action ---
            action = {
                "shoulder_pan.pos": target_joints[0],
                "shoulder_lift.pos": target_joints[1],
                "elbow_flex.pos": target_joints[2],
                "wrist_flex.pos": target_joints[3],
                "wrist_roll.pos": target_joints[4],
                "gripper.pos": target_gripper,
            }
            
            robot.send_action(action)
            time.sleep(1 / 30.0)
            
    except KeyboardInterrupt:
        print("\nDisconnecting and safely disabling torque...")
        robot.disconnect()
        pygame.quit()

if __name__ == "__main__":
    main() 
