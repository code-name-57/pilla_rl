import argparse
import os
import pickle
from importlib import metadata
import torch
import time

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.3.3":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
import sys
import select
import termios
import tty


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-standup")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # Commands for standup task (no movement commands needed)
    lin_x = 0.0
    lin_y = 0.0
    ang_z = 0.0

    print("Starting standup evaluation...")
    print("The robot will start from random poses and attempt to stand up.")
    print("Press Ctrl+C to exit.")


    obs, _ = env.reset()
    warmup_steps = int(2.0 / env.dt)  # 2 seconds of simulation
    for _ in range(warmup_steps):
        env.commands = torch.tensor([[lin_x, lin_y, ang_z]], dtype=torch.float).to("cuda:0").repeat(1, 1)
        obs, rews, dones, infos = env.step(torch.zeros((1, env.num_actions), device=gs.device))
        time.sleep(env.dt)

    with torch.no_grad():
        while True:
            actions = policy(obs)
            env.commands = torch.tensor([[lin_x, lin_y, ang_z]], dtype=torch.float).to("cuda:0").repeat(1, 1)
            obs, rews, dones, infos = env.step(actions)


if __name__ == "__main__":
    main()

"""
# evaluation
python go2_eval.py -e go2-standup --ckpt 100
"""
