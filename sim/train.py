"""Train PPO on privileged simulator state; use --steps 512 for a pipeline smoke test."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from sim.env import CubeLiftEnv, MODEL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=1024)
    parser.add_argument("--start", choices=["near", "home"], default="near")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--out", type=Path, default=Path("runs/cube_lift"))
    parser.add_argument("--resume", type=Path, help="PPO checkpoint to continue; use a new output directory")
    args = parser.parse_args()
    if args.steps <= 0 or args.envs <= 0 or args.rollout <= 1:
        parser.error("steps/envs must be positive and rollout must exceed 1")
    if args.envs * args.rollout % 64:
        parser.error("envs * rollout must be divisible by batch size 64")
    args.out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    packages = ("mujoco", "gymnasium", "stable-baselines3", "torch", "numpy")
    config["versions"] = {p: version(p) for p in packages}
    config["model_sha256"] = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    config["environment_sha256"] = hashlib.sha256(Path(__file__).with_name("env.py").read_bytes()).hexdigest()
    config["observation"] = "privileged simulator state; see sim/README.md"
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.set_num_threads(1)
    train_env = make_vec_env(CubeLiftEnv, n_envs=args.envs, seed=args.seed,
                             env_kwargs={"start": args.start}, monitor_dir=str(args.out / "monitor"))
    eval_env = make_vec_env(CubeLiftEnv, n_envs=1, seed=10_000,
                            env_kwargs={"start": args.start})
    try:
        if args.resume:
            model = PPO.load(args.resume, env=train_env, device=args.device)
            if model.n_steps != args.rollout:
                parser.error("For resume, --rollout must match the saved model's n_steps")
            model.set_random_seed(args.seed)
        else:
            model = PPO("MlpPolicy", train_env, device=args.device, seed=args.seed, verbose=1,
                        n_steps=args.rollout, batch_size=64, learning_rate=3e-4,
                        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}})
        callbacks = [
            CheckpointCallback(
                save_freq=max(50_000 // args.envs, 1), save_path=str(args.out / "checkpoints")
            ),
            EvalCallback(eval_env, eval_freq=max(20_000 // args.envs, 1), n_eval_episodes=10,
                         best_model_save_path=str(args.out / "best"), log_path=str(args.out / "evaluation")),
        ]
        try:
            model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=not args.resume)
        finally:
            model.save(args.out / "last")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
