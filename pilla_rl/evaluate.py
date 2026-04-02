"""Unified evaluation entry point for all pilla_rl tasks.

Usage::

    python -m pilla_rl.evaluate \\
        --config pilla_rl/configs/tasks/walk.yaml \\
        --checkpoint logs/go2-walking/model_5000.pt

Dependencies: rsl-rl-lib==2.3.3, pyyaml
"""

import argparse
import os
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

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from pilla_rl.config_loader import load_task_config, instantiate_env


def main():
    parser = argparse.ArgumentParser(description="pilla_rl unified evaluation script")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML task config file (e.g. pilla_rl/configs/tasks/walk.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (default: 1)")
    args = parser.parse_args()

    gs.init(logging_level="warning")

    # ------------------------------------------------------------------
    # Load config — disable rewards during evaluation
    # ------------------------------------------------------------------
    config = load_task_config(args.config)
    config["reward_cfg"]["reward_scales"] = {}

    # ------------------------------------------------------------------
    # Create environment with viewer
    # ------------------------------------------------------------------
    env = instantiate_env(config, num_envs=args.num_envs, show_viewer=True)

    # ------------------------------------------------------------------
    # Load policy
    # ------------------------------------------------------------------
    exp_name = config.get("train", {}).get("exp_name", "eval_run")
    log_dir = os.path.join("logs", exp_name + "_eval")
    os.makedirs(log_dir, exist_ok=True)

    train_cfg = config.get("train", {})
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["resume"] = True
    train_cfg["runner"]["resume_path"] = args.checkpoint

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=gs.device)

    # ------------------------------------------------------------------
    # Inference loop
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    main()
