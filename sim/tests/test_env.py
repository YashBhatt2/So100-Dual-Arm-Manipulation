import unittest

import mujoco
import numpy as np
from stable_baselines3.common.env_checker import check_env

from sim.demo import run_demo
from sim.env import CubeLiftEnv


class CubeLiftTests(unittest.TestCase):
    def setUp(self):
        self.env = CubeLiftEnv()

    def tearDown(self):
        self.env.close()

    def test_gym_api(self):
        check_env(self.env)

    def test_seed_reproduces_reset_and_step(self):
        a, _ = self.env.reset(seed=42)
        a1 = self.env.step(np.zeros(4))[0]
        b, _ = self.env.reset(seed=42)
        b1 = self.env.step(np.zeros(4))[0]
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a1, b1)

    def test_airborne_object_without_grasp_is_not_success(self):
        self.env.reset(seed=0)
        self.env.data.qpos[self.env.cube_qadr:self.env.cube_qadr + 3] = [0.12, -0.3, 0.2]
        mujoco.mj_forward(self.env.model, self.env.data)
        for _ in range(14):
            _, _, _, _, info = self.env.step([0, 0, 0, 1])
            self.assertFalse(info["is_success"])
            self.assertEqual(info["hold_steps"], 0)

    def test_timeout_and_reset_required(self):
        self.env.max_steps = 1
        self.env.reset(seed=0)
        _, _, terminated, truncated, _ = self.env.step(np.zeros(4))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(4))

    def test_bad_action_rejected(self):
        self.env.reset(seed=0)
        with self.assertRaises(ValueError):
            self.env.step([np.nan, 0, 0, 0])

    def test_scripted_grasp_lifts_real_free_object(self):
        self.assertEqual(self.env.model.neq, 0)  # No hidden attachment/weld constraints.
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                result = run_demo(seed)
                self.assertTrue(result["is_success"], result)
                self.assertGreaterEqual(result["lift_height"], 0.06)
                self.assertGreaterEqual(result["hold_steps"], 13)


if __name__ == "__main__":
    unittest.main()
