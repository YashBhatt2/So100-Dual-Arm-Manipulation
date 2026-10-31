"""Evaluate a saved policy on explicit held-out episode seeds."""
import argparse
import json
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO

from sim.env import CubeLiftEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--start", choices=["near", "home"], default="near")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    model = PPO.load(args.checkpoint, device="cpu")
    env = CubeLiftEnv(start=args.start, render_mode="human" if args.render else None)
    episodes = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            obs, _ = env.reset(seed=seed)
            total_reward, max_lift = 0.0, 0.0
            while True:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                max_lift = max(max_lift, info["lift_height"])
                if args.render:
                    time.sleep(env.dt)
                if terminated or truncated:
                    episodes.append({"seed": seed, "success": info["is_success"], "steps": env.steps,
                                     "reward": total_reward, "max_lift": max_lift})
                    break
    finally:
        env.close()
    report = {"checkpoint": str(args.checkpoint), "start": args.start,
              "success_rate": sum(e["success"] for e in episodes) / args.episodes, "episodes": episodes}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
