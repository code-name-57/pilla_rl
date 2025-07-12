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
            "entropy_coef": 0.01,   # Higher entropy for more exploration in complex recovery
            "gamma": 0.998,         # Slightly higher gamma for longer horizon planning
            "lam": 0.95,
            "learning_rate": 0.0003, # Slightly lower learning rate for stability
            "max_grad_norm": 1.0,
            "num_learning_epochs": 10,  # More epochs per update for complex task
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
        "num_actions": 12,
        # joint/link names
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination - be more lenient during recovery learning
        "termination_if_roll_greater_than": 180,  # Don't terminate on roll (allow upside-down)
        "termination_if_pitch_greater_than": 90,  # More lenient on pitch
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,  # Longer episodes for recovery task
        "resampling_time_s": 10.0,  # Less frequent command resampling
        "action_scale": 0.3,        # Slightly larger action scale for recovery movements
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 48,  # Updated from 45 to 48 (added 3 for base orientation)
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.42,  # Target standing height
        "feet_height_target": 0.075,
        "reward_scales": {
            # Remove locomotion rewards - focus on recovery and standup
            "tracking_lin_vel": 0.0,
            "tracking_ang_vel": 0.0,
            
            # Basic penalties (kept low to not interfere with main objectives)
            "lin_vel_z": -1.0,              # Light penalty for vertical movement
            "base_height": -1.0,            # Reduced penalty for wrong height (now we have positive height rewards)
            "action_rate": -0.02,           # Penalize rapid actions slightly
            "similar_to_default": -0.1,     # Very light penalty for joint deviation
            
            # Phase 1: Recovery from upside-down position
            "upright_orientation": 15.0,    # Strong reward for upright posture
            "recovery_progress": 25.0,      # Main reward for recovery progress (staged orientation+height)
            "minimize_base_roll": 12.0,     # Focus on roll recovery (key for upside-down)
            
            # Phase 2: Standing up once upright (these are the key additions)
            "standup_height": 20.0,         # Strong reward for proper height when upright
            "complete_standup": 50.0,       # Huge bonus for achieving full standup (orientation + height + stability)
            "height_when_upright": 30.0,    # Progressive height reward that scales with orientation quality
            
            # Supporting rewards
            "stability": 5.0,               # Reward controlled movement
            "legs_not_in_air": 8.0,         # Encourage grounded, controlled leg movement
            "energy_efficiency": 2.0,       # Encourage efficient movements (reduced to not interfere)
            "forward_progress": 1.0,        # Small reward for forward motion when upright (reduced)
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0, 0],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-upside-down-recovery")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=20000)  # More iterations for complex recovery task
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
# training for upside-down recovery
python go2_train.py -e go2-upside-down-recovery
"""
