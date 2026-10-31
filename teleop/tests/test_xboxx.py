"""Hardware-free regression checks for teleoperation control math."""
import unittest

import numpy as np

from xboxx import JOINTS, SimulatedArm, compute_action, deadzone


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.arm = SimulatedArm()

    def action(self, inputs, dt=1 / 30):
        return compute_action(self.arm.get_observation(), self.arm, inputs, dt)

    def test_released_enable_holds_measured_pose(self):
        self.assertEqual(self.action({"x": 1, "grip": 1}), self.arm.get_observation())

    def test_translation_uses_transform_and_current_seed(self):
        action = self.action({"enabled": True, "x": 0.5})
        self.assertAlmostEqual(action["shoulder_pan.pos"], 5 / 6)
        self.assertEqual(action["shoulder_lift.pos"], 0)

    def test_unreachable_targets_do_not_accumulate(self):
        for _ in range(100):
            action = self.action({"enabled": True, "x": 1})
            self.assertEqual(action["shoulder_pan.pos"], 1)

    def test_roll_and_gripper_units_and_bounds(self):
        self.arm.observation["gripper.pos"] = 99.8
        action = self.action({"enabled": True, "roll": -1, "grip": 1})
        self.assertEqual(action["wrist_roll.pos"], -1)
        self.assertEqual(action["gripper.pos"], 100)

    def test_invalid_ik_stops_instead_of_sending(self):
        self.arm.inverse_kinematics = lambda *args: np.full(5, np.nan)
        with self.assertRaises(ValueError):
            self.action({"enabled": True, "x": 1})

    def test_missing_observation_is_not_replaced_with_zero(self):
        del self.arm.observation[f"{JOINTS[0]}.pos"]
        with self.assertRaises(KeyError):
            self.action({})

    def test_deadzone(self):
        self.assertEqual(deadzone(0.1), 0)
        self.assertEqual(deadzone(-1), -1)
        self.assertEqual(deadzone(1), 1)


if __name__ == "__main__":
    unittest.main()
