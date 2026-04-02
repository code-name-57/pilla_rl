"""Unified training entry point for all pilla_rl tasks.

Usage::

    # Walk task with default settings
    python -m pilla_rl.train --config pilla_rl/configs/tasks/walk.yaml

    # Standup task with custom num_envs
    python -m pilla_rl.train --config pilla_rl/configs/tasks/standup.yaml --num_envs 2048

    # Transfer learning from a walk checkpoint
    python -m pilla_rl.train \\
        --config pilla_rl/configs/tasks/standup.yaml \\
        --resume_from logs/go2-walking/model_5000.pt

Dependencies: rsl-rl-lib==2.3.3, pyyaml
"""

import argparse
import os
import pickle
import shutil
import copy
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.3.3":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.3.3'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from pilla_rl.config_loader import load_task_config, instantiate_env


def _build_train_cfg(config: dict, exp_name: str, max_iterations: int) -> dict:
    """Construct the rsl-rl train config dict from the YAML ``train`` section."""
    train = copy.deepcopy(config.get("train", {}))

    # Override experiment name and max_iterations if provided
    train.setdefault("runner", {})
    train["runner"]["experiment_name"] = exp_name
    train["runner"]["max_iterations"] = max_iterations

    return train


def main():
    parser = argparse.ArgumentParser(description="pilla_rl unified training script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML task config file (e.g. pilla_rl/configs/tasks/walk.yaml)",
    )
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override max training iterations (defaults to value in config)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint (.pt) for transfer learning / curriculum continuation",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Override experiment name (defaults to value in config train section)",
    )
    args = parser.parse_args()

    gs.init(logging_level="warning")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config = load_task_config(args.config)

    exp_name = args.exp_name or config.get("train", {}).get("exp_name", "pilla_rl_run")
    max_iterations = args.max_iterations or config.get("train", {}).get("runner", {}).get("max_iterations", 10000)

    log_dir = os.path.join("logs", exp_name)

    # ------------------------------------------------------------------
    # Build and save configs for backward compatibility with eval scripts
    # ------------------------------------------------------------------
    env_cfg = config["env_cfg"]
    obs_cfg = config["obs_cfg"]
    reward_cfg = config["reward_cfg"]
    command_cfg = config["command_cfg"]
    train_cfg = _build_train_cfg(config, exp_name, max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(os.path.join(log_dir, "cfgs.pkl"), "wb"),
    )

    # ------------------------------------------------------------------
    # Create environment
    # ------------------------------------------------------------------
    env = instantiate_env(config, num_envs=args.num_envs, show_viewer=False)

    # ------------------------------------------------------------------
    # Runner setup — optional checkpoint resume
    # ------------------------------------------------------------------
    if args.resume_from is not None:
        train_cfg.setdefault("runner", {})
        train_cfg["runner"]["resume"] = True
        train_cfg["runner"]["resume_path"] = args.resume_from

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
