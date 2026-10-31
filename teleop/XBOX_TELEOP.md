# Xbox SO-100 teleoperation

## Check without hardware

With NumPy installed, run:

```sh
python3 xboxx.py --dry-run --demo --steps 120
python3 -m unittest discover -s tests -v
```

The demo uses a simple deterministic loopback model. It checks control flow and
command math, not SO-100 reachability, URDF correctness, or physical dynamics.

## Use a controller without the arm

Install `pygame>=2.5,<3` in your Python environment, connect an Xbox-compatible
controller, then run `python xboxx.py --dry-run`. SDL's game-controller mapping
normalizes supported devices instead of relying on platform-specific axis numbers.

Hold **X** to enable movement. Release X to command the measured position.
Left stick controls X/Y, triggers control Z, bumpers rotate the wrist, and A/B
close/open the gripper. Start or Ctrl+C exits. Disconnecting the controller exits.

## Connect the robot

Use the bundled LeRobot environment, including its `feetech` and `kinematics`
extras (see `lerobot/pyproject.toml`), and install pygame there too. Calibrate the
arm using LeRobot before running this script. Supply the existing calibration ID,
serial port, matching SO-100 URDF, and the actual end-effector frame in that URDF:

```sh
python xboxx.py --port /dev/ttyACM0 --robot-id my_arm \
  --urdf /absolute/path/to/so100.urdf --frame gripper_frame_link
```

The five URDF joint names must match `JOINTS` in `xboxx.py`. No URDF is included
in this repository. Cartesian motion uses metres in the URDF world frame; arm
joints use degrees and gripper position uses 0–100 percent. Defaults limit
translation to 0.05 m/s, joints to 30 degrees/s, and gripper motion to 30 percent/s.
Targets restart from measured feedback each tick to avoid accumulated commands.
The robot backend also limits relative commands to 2 units per motor update.

Verify frame directions with small inputs and the arm clear of obstacles on the
first hardware run. Disconnect disables torque, so support the arm when stopping.
The script does not perform collision checking or certify physical joint limits.
