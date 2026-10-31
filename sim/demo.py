"""Scripted physical grasp check; no policy training or object attachment."""
import argparse
import json
import time

import numpy as np

from sim.env import CubeLiftEnv


def run_demo(seed=0, render=False):
    env = CubeLiftEnv(render_mode="human" if render else None)
    try:
        env.reset(seed=seed)
        xy = env.cube_pos[:2].copy()
        max_lift = 0.0
        for step in range(env.max_steps):
            target = np.r_[xy, 0.014 if step < 130 else 0.14]
            action = np.r_[np.clip((target - env.tip) / 0.004, -1, 1), 1 if step < 90 else -1]
            _, _, terminated, truncated, info = env.step(action)
            max_lift = max(max_lift, info["lift_height"])
            if render:
                time.sleep(env.dt)
            if terminated or truncated:
                return {"seed": seed, "steps": step + 1, "max_lift": max_lift, **info}
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    result = run_demo(args.seed, args.render)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["is_success"] else 1)
