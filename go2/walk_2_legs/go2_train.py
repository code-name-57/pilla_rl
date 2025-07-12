import argparse
import os
import pickle
import shutil
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

from go2_env import Go2Env


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "logger": "tensorboard"
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,  # Only front 2 legs: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": -0.5,  # Rear legs positioned for stability
            "RR_hip_joint": -0.5,
            "FL_thigh_joint": -0.8,  # Inverted front leg position
            "FR_thigh_joint": -0.8,  # Inverted front leg position
            "RL_thigh_joint": 2.0,  # Rear legs bent to lift them off ground
            "RR_thigh_joint": 2.0,
            "FL_calf_joint": 1.5,   # Inverted front leg position
            "FR_calf_joint": 1.5,   # Inverted front leg position
            "RL_calf_joint": -2.5,  # Rear legs folded up
            "RR_calf_joint": -2.5,
        },
        "joint_names": [
            # Only front legs are controllable
            "FL_hip_joint",
            "FL_thigh_joint", 
            "FL_calf_joint",
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
        ],
        # PD
        "kp": 25.0,  # Slightly higher for bipedal stability
        "kd": 0.8,
        # termination
        "termination_if_roll_greater_than": 15,  # More lenient for bipedal
        "termination_if_pitch_greater_than": 15,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.5],  # Higher initial position for bipedal stance
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 15.0,  # Shorter episodes initially
        "resampling_time_s": 3.0,
        "action_scale": 0.3,  # Slightly larger action scale
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 33,  # Reduced from 45: 3(ang_vel) + 3(gravity) + 3(commands) + 6(dof_pos) + 6(dof_vel) + 6(actions) + 6(rear_leg_pos)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.3,
        "base_height_target": 0.4,  # Target height for inverted bipedal stance
        "feet_height_target": 0.1,
        "reward_scales": {
            "tracking_lin_vel": 1.0,  # Reduced for stability focus
            "tracking_ang_vel": 0.5,  # Reduced for stability focus
            "lin_vel_z": -2.0,
            "base_height": -20.0,  # Reduced penalty
            "action_rate": -0.01,
            "similar_to_default": -0.05,
            "bipedal_stability": 15.0,  # Increased reward for inverted stability
            "foot_contact": 3.0,  # Increased reward for proper foot contact
            "torso_upright": 10.0,  # Increased reward for inverted torso orientation
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [-0.5, 1.0],  # Reduced speed range for bipedal
        "lin_vel_y_range": [-0.3, 0.3],
        "ang_vel_range": [-0.3, 0.3],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python examples/locomotion/go2_train.py
"""
