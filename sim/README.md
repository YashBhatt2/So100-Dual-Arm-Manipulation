# SO-100 cube lifting with privileged state

This learning exercise trains one arm to lift a 20 mm, 20 g cube using exact MuJoCo state. No camera, segmentation model, LeRobot installation, robot connection, or domain randomization is needed. The task is lifting, not placing or general manipulation. The original downloaded URDF is not used because its meshes and jaw collisions were missing.

## Setup on Linux

Run all commands from the repository root. Create an isolated Python 3.12 environment:

```sh
python3.12 -m venv .venv-sim
source .venv-sim/bin/activate
python -m pip install -r sim/requirements.txt
wandb login  # paste your API key; stored in your local user config
python -m unittest discover -s sim/tests -v
python -m sim.demo
python -m sim.demo --render
```

The model and meshes are included, so cloning this repository is sufficient. Training and non-rendered evaluation do not require a display. Mac development uses the same environment; for a visible viewer on macOS use `mjpython -m sim.demo --render` instead of `python`. Headless Linux offscreen rendering may require an appropriate OpenGL backend, but training does not render.

## What the policy learns

The 32 observation values are six joint positions, six joint velocities, gripper reference position (3), cube position/quaternion (7), cube angular/linear velocity (6), cube-to-gripper displacement (3), and normalized consecutive successful hold time (1). Distances are metres and joint angles are radians. All are simulator measurements, including information not directly available on a physical robot.

Four bounded actions command XYZ velocity (maximum 0.1 m/s requested) and jaw position, where -1 closes and +1 opens. A damped-Jacobian IK controller handles the five arm joints and tries to keep the gripper pointing down. Joint target increments are limited. Actual motion may be slower than the requested velocity due to dynamics and controller tracking. This is learning manipulation above an IK controller, not learning motor control from scratch.

The control rate is 25 Hz and physics runs at 500 Hz. The cube must rise at least 6 cm above its initial resting height while contacting both jaw bodies for 13 consecutive control steps (0.52 seconds). Tossing it into the air does not count. Episodes time out after 500 steps (20 simulated seconds). There are no attachment/weld constraints or scripted actions inside the training environment.

The reward logic is in `sim/env.py`: inspect `REWARD_CONFIG`, `_potential_terms()`, and `_reward()`. Each step reports its individual components in `info["reward_components"]`; their `total` is the exact reward PPO receives. The components are approach progress (weight 1; closeness shaped by `tanh(20 × distance)`), valid two-jaw grasp progress (weight 1), lift progress (weight 4, scaled to 6 cm and allowed only while both jaws touch), success (+10 after 13 consecutive lifted/contact steps), workspace failure (-2), time (-0.01 per 0.04-second action), and XYZ action size (-0.001 times squared action magnitude). `REWARD_DISCOUNT` is 0.99 and is also passed as PPO's `gamma`. Progress terms are discounted changes in potential, `gamma × next - previous`; on terminal success/failure, next potential is zero. The named values and formulas are stored in the W&B run config, and separate per-step component averages are charted under `reward_terms/`. These weights are initial choices for learning, not tuned or scientifically validated values.

By default `--start near` initializes the arm 8 cm above the cube with its jaw open. The object starts on the floor, never pre-grasped. Cube X/Y varies slightly each episode. This deliberately easy distribution reduces initial exploration difficulty. `--start home` uses one fixed elevated arm start, requiring more approach movement. Neither setting is domain randomization of physics or visual appearance.

## Training and inspecting results

First check the complete train/save/load pipeline:

```sh
python -m sim.train --steps 512 --rollout 128 --out runs/smoke
python -m sim.evaluate runs/smoke/last.zip --episodes 2
```

A smoke run is not enough to learn a grasp. Training automatically creates a W&B run in the `so100-cube-lift` project and syncs PPO/TensorBoard scalars such as rollout reward, episode length, success rate, each reward term under `reward_terms/`, value/policy losses, entropy, and periodic evaluation success/reward. Open the run URL printed by W&B to view charts. Override the project with `--wandb-project my-project`; use `--wandb-entity TEAM_OR_USER` for a team. W&B is required by the training command. If the Linux machine is temporarily offline, set `WANDB_MODE=offline` and later run `wandb sync` on the saved offline run directory.

Start a longer experiment with:

```sh
python -m sim.train --steps 1000000 --seed 0 --out runs/cube_seed0
python -m sim.evaluate runs/cube_seed0/last.zip --episodes 100 \
  --report runs/cube_seed0/test_results.json
python -m sim.evaluate runs/cube_seed0/last.zip --episodes 5 --render
```

One million transitions is an initial budget, not a promise of convergence. PPO rounds training to complete rollouts. If success remains low, inspect videos, reaching behavior, contact frequency and lift height before increasing the budget. Compare against the scripted demo to distinguish learning problems from impossible grasps.

CPU is the default for this small MLP policy. `--envs` controls the number of environments collected in one process; these are not parallel physics workers. `--device cuda` or `mps` is available but may be slower for a small network. Benchmark on the training machine.

Outputs include configuration and package versions, model/environment hashes, the W&B run URL, TensorBoard event files, episode monitor CSVs, periodic checkpoints, development evaluation arrays, and `last.zip`. W&B logs optimization and rollout metrics; use the existing evaluation JSON/NPZ and Monitor CSVs for per-episode success details. The development evaluator selects `best/best_model.zip` by mean reward; separately measure success when comparing policies. Development evaluation uses seed 10000; final evaluation defaults to seeds starting at 20000. Never tune using the final evaluation set. Repeat complete training with different seeds to assess reliability.

Continue a run into a new output directory:

```sh
python -m sim.train --resume runs/cube_seed0/last.zip --steps 1000000 \
  --out runs/cube_seed0_continued
```

Continuation loads policy/optimizer state but resets environments and reseeds randomness; it is not bit-for-bit continuation. Preserve the environment code and use the original rollout length (default 1024). For the 128-step smoke checkpoint, pass `--rollout 128`. Changing `--start` is a deliberate curriculum change; document it and evaluate both start distributions separately. Existing output directories are refused to protect results. Ctrl+C during learning saves the latest model in `last.zip`.

## Model provenance and limitations

`models/so100/` contains the SO-ARM100 v1.3 model from [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trs_so_arm100), with its Apache-2.0 license and original README. `UPSTREAM_COMMIT` pins the fetched revision. The upstream model is kept intact; `env.py` adds the cube/floor/site, removes incompatible keyframes, and selects timestep/integration options in memory.

This model has approximate joint limits and actuator/contact parameters. Its joint conventions differ from the standalone URDF in the root. A simulation checkpoint is not ready to send to the real robot. Later transfer needs calibrated joint mappings, measured actuator behavior, observation estimation, and tests under plausible dynamics/perception variation. Segmentation alone does not provide all policy observations. The exact ETH model/paper referred to as “Defm” still needs identification before implementing that perception stage.

## Validation status

The scripted controller physically grasped, lifted, and held the cube at seeds 0, 1, and 2. Regression checks cover the Gymnasium API, deterministic seeding, timeout behavior, invalid actions, and rejection of airborne non-grasped objects. Short PPO train/save/load/continuation runs validate the software pipeline. These checks do not establish a trained policy's success rate or real-world transfer.
