# Repository Guidelines

## Project Structure & Module Organization

This repository supports ARC research on SO-100 single-arm control and coordinated dual-arm manipulation.

- `xboxx.py`: Xbox controller teleoperation using LeRobot and inverse kinematics.
- `vision/`: RealSense camera experiments, ArUco detection, `marker.png`, and camera dependencies.
- `lerobot/`: bundled LeRobot project; library code lives in `src/lerobot/`, tests in `tests/`, tutorials in `examples/`, and documentation in `docs/source/`.
- `papers/`, `README.md`, and research/troubleshooting documents: background material and experiment plans.

Read `lerobot/AGENTS.md` and `lerobot/CONTRIBUTING.md` before changing that subtree.

## Build, Test, and Development Commands

Run LeRobot commands from `lerobot/`:

- `uv sync --locked --extra test --extra dev`: install the locked development and test environment (Python 3.12+).
- `uv run pytest tests -svv --maxfail=10`: run the test suite; some tests need optional dependencies, artifacts, or hardware.
- `uv run pre-commit run --all-files`: run configured formatting, lint, and other quality checks.
- `DEVICE=cpu make test-end-to-end`: exercise policy training and evaluation; requires relevant extras and dataset access.

From the repository root, `python vision/aruco_realsense.py` runs marker detection and `python xboxx.py` starts teleoperation in a suitably configured environment. Camera dependencies are listed in `vision/requirements.txt`; teleoperation also imports `pygame`. Verify compatibility between that file's pinned LeRobot version and the bundled checkout.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` functions/modules, `PascalCase` classes, and `UPPER_CASE` configuration constants. Follow LeRobot's Ruff configuration: 110-character lines, double quotes, sorted imports, and Google-style docstrings. Add type annotations when extending library interfaces. Give new scripts descriptive names without spaces.

## Testing Guidelines

LeRobot uses pytest; name tests `test_*.py` and place them beside related tests under `lerobot/tests/`. Run affected tests before the broader suite. No repository-wide coverage threshold is specified. Root and vision scripts have no dedicated automated suite; document camera, controller, and robot smoke-test results for changes affecting them.

## Commit & Pull Request Guidelines

Root history contains descriptive prose and terse messages, with no established commit prefix convention. Use concise imperative subjects describing the change. PRs should explain purpose, link relevant issues, list validation commands/results, and include screenshots for visual changes or hardware observations for control changes.

## Hardware Configuration

Check `PORT`, `URDF_PATH`, calibration, and controller mappings before teleoperation. Keep machine-specific paths, credentials, recordings, and model checkpoints out of commits.
