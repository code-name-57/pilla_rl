"""Centralized reward function library for pilla_rl.

All reward functions are standalone pure functions that accept explicit tensor
arguments rather than accessing environment state via ``self``.  Each function
returns a per-environment reward tensor of shape ``(num_envs,)``.

The ``REWARD_REGISTRY`` dict maps string names to functions so that
``BaseQuadrupedEnv`` can look up and call them by name.
"""

import torch


# ---------------------------------------------------------------------------
# Basic locomotion rewards (from go2/walk/go2_env.py)
# ---------------------------------------------------------------------------


def tracking_lin_vel(commands, base_lin_vel, tracking_sigma):
    """Tracking of linear velocity commands (xy axes)."""
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def tracking_ang_vel(commands, base_ang_vel, tracking_sigma):
    """Tracking of angular velocity commands (yaw)."""
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def lin_vel_z(base_lin_vel):
    """Penalize z axis base linear velocity."""
    return torch.square(base_lin_vel[:, 2])


def action_rate(last_actions, actions):
    """Penalize changes in actions."""
    return torch.sum(torch.square(last_actions - actions), dim=1)


def similar_to_default(dof_pos, default_dof_pos):
    """Penalize joint poses far away from default pose."""
    return torch.sum(torch.abs(dof_pos - default_dof_pos), dim=1)


def base_height(base_pos, base_height_target):
    """Penalize base height away from target."""
    return torch.square(base_pos[:, 2] - base_height_target)


# ---------------------------------------------------------------------------
# Standup rewards (from go2/standup_copilot/go2_env.py)
# ---------------------------------------------------------------------------


def upright_orientation(base_euler):
    """Reward for maintaining upright orientation (small roll and pitch).

    Combines a sharp peak at upright with a gradual component that provides
    gradients even when the robot is far from upright.

    Source: go2/standup_copilot/go2_env.py
    """
    roll_pitch_penalty = torch.square(base_euler[:, 0] * torch.pi / 180) + torch.square(
        base_euler[:, 1] * torch.pi / 180
    )
    sharp_reward = torch.exp(-roll_pitch_penalty / 0.1)
    gradual_reward = torch.exp(-roll_pitch_penalty / 2.0)
    return 0.7 * sharp_reward + 0.3 * gradual_reward


def stability(base_ang_vel):
    """Reward for low angular velocity (stability).

    Source: go2/standup_copilot/go2_env.py
    """
    ang_vel_penalty = torch.sum(torch.square(base_ang_vel), dim=1)
    return torch.exp(-ang_vel_penalty / 0.5)


def stand_up_progress(base_pos, base_euler, base_height_target):
    """Progressive reward for standing up — combines height and orientation.

    Source: go2/standup_copilot/go2_env.py
    """
    height_reward = torch.exp(-torch.square(base_pos[:, 2] - base_height_target) / 0.1)
    roll_pitch_penalty = torch.square(base_euler[:, 0] * torch.pi / 180) + torch.square(
        base_euler[:, 1] * torch.pi / 180
    )
    sharp_orientation = torch.exp(-roll_pitch_penalty / 0.1)
    gradual_orientation = torch.exp(-roll_pitch_penalty / 2.0)
    orientation_reward = 0.7 * sharp_orientation + 0.3 * gradual_orientation
    return height_reward * orientation_reward


def recovery_effort(base_euler, dof_vel):
    """Encourage movement when far from upright to prevent the policy from
    getting stuck.

    Source: go2/standup_copilot/go2_env.py
    """
    roll_pitch_penalty = torch.square(base_euler[:, 0] * torch.pi / 180) + torch.square(
        base_euler[:, 1] * torch.pi / 180
    )
    is_far_from_upright = roll_pitch_penalty > (45 * torch.pi / 180) ** 2
    joint_movement = torch.sum(torch.square(dof_vel), dim=1)
    movement_reward = torch.exp(-joint_movement / 10.0)
    return torch.where(is_far_from_upright, 1.0 - movement_reward, torch.zeros_like(movement_reward))


def joint_regularization(dof_pos, default_dof_pos):
    """Encourage joint positions closer to default for stability.

    Source: go2/standup_copilot/go2_env.py
    """
    joint_deviation = torch.sum(torch.square(dof_pos - default_dof_pos), dim=1)
    return torch.exp(-joint_deviation / 2.0)


# ---------------------------------------------------------------------------
# Recovery rewards (from go2/upside_down_recovery/go2_env.py)
# ---------------------------------------------------------------------------


def recovery_progress(base_euler, base_pos, base_height_target):
    """Main reward for upside-down recovery progress.

    Source: go2/upside_down_recovery/go2_env.py
    """
    roll_rad = base_euler[:, 0] * torch.pi / 180
    pitch_rad = base_euler[:, 1] * torch.pi / 180
    roll_progress = (torch.cos(roll_rad) + 1.0) / 2.0
    pitch_progress = (torch.cos(pitch_rad) + 1.0) / 2.0
    orientation_progress = roll_progress * pitch_progress
    height_factor = torch.exp(-torch.square(base_pos[:, 2] - base_height_target) / 0.2)
    return orientation_progress * (0.7 + 0.3 * height_factor)


def legs_not_in_air(dof_vel):
    """Encourage legs to be grounded rather than flailing.

    Source: go2/upside_down_recovery/go2_env.py
    """
    joint_vel_penalty = torch.sum(torch.square(dof_vel), dim=1)
    return torch.exp(-joint_vel_penalty / 20.0)


def energy_efficiency(actions):
    """Encourage efficient movements — penalize excessive joint torques.

    Source: go2/upside_down_recovery/go2_env.py
    """
    action_magnitude = torch.sum(torch.square(actions), dim=1)
    return torch.exp(-action_magnitude / 5.0)


def forward_progress(base_euler, base_lin_vel):
    """Small reward for forward movement during recovery (only when reasonably
    upright).

    Source: go2/upside_down_recovery/go2_env.py
    """
    roll_rad = base_euler[:, 0] * torch.pi / 180
    is_reasonably_upright = torch.abs(roll_rad) < (torch.pi / 2)
    forward_vel = base_lin_vel[:, 0]
    forward_reward = torch.clamp(forward_vel, 0.0, 1.0)
    return torch.where(is_reasonably_upright, forward_reward, torch.zeros_like(forward_reward))


def minimize_base_roll(base_euler):
    """Reward for minimizing roll angle — key for upside-down recovery.

    Source: go2/upside_down_recovery/go2_env.py
    """
    roll_rad = torch.abs(base_euler[:, 0] * torch.pi / 180)
    return torch.exp(-roll_rad / (torch.pi / 4))


# ---------------------------------------------------------------------------
# Combined upside-down standup rewards (from go2/upside_down_standup/go2_env.py)
# ---------------------------------------------------------------------------


def standup_height(base_euler, base_pos, base_height_target):
    """Strong reward for achieving proper standing height once upright.

    Source: go2/upside_down_standup/go2_env.py
    """
    roll_rad = torch.abs(base_euler[:, 0] * torch.pi / 180)
    pitch_rad = torch.abs(base_euler[:, 1] * torch.pi / 180)
    is_upright = (roll_rad < torch.pi / 6) & (pitch_rad < torch.pi / 6)
    height_error = torch.abs(base_pos[:, 2] - base_height_target)
    height_reward = torch.exp(-height_error / 0.1)
    return torch.where(is_upright, height_reward, torch.zeros_like(height_reward))


def complete_standup(base_euler, base_pos, base_ang_vel, base_height_target):
    """Bonus reward for achieving both upright orientation AND proper height.

    Source: go2/upside_down_standup/go2_env.py
    """
    roll_rad = torch.abs(base_euler[:, 0] * torch.pi / 180)
    pitch_rad = torch.abs(base_euler[:, 1] * torch.pi / 180)
    height_error = torch.abs(base_pos[:, 2] - base_height_target)
    is_orientation_good = (roll_rad < torch.pi / 12) & (pitch_rad < torch.pi / 12)
    is_height_good = height_error < 0.05
    is_stable = torch.sum(torch.square(base_ang_vel), dim=1) < 2.0
    is_fully_standup = is_orientation_good & is_height_good & is_stable
    return torch.where(is_fully_standup, torch.ones_like(roll_rad), torch.zeros_like(roll_rad))


def height_when_upright(base_euler, base_pos, base_height_target):
    """Progressive height reward that increases as orientation improves.

    Source: go2/upside_down_standup/go2_env.py
    """
    roll_rad = torch.abs(base_euler[:, 0] * torch.pi / 180)
    pitch_rad = torch.abs(base_euler[:, 1] * torch.pi / 180)
    roll_quality = torch.exp(-roll_rad / (torch.pi / 6))
    pitch_quality = torch.exp(-pitch_rad / (torch.pi / 6))
    orientation_quality = roll_quality * pitch_quality
    height_error = torch.abs(base_pos[:, 2] - base_height_target)
    height_quality = torch.exp(-height_error / 0.08)
    return orientation_quality * height_quality


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REWARD_REGISTRY: dict = {
    # walk / locomotion
    "tracking_lin_vel": tracking_lin_vel,
    "tracking_ang_vel": tracking_ang_vel,
    "lin_vel_z": lin_vel_z,
    "action_rate": action_rate,
    "similar_to_default": similar_to_default,
    "base_height": base_height,
    # standup
    "upright_orientation": upright_orientation,
    "stability": stability,
    "stand_up_progress": stand_up_progress,
    "recovery_effort": recovery_effort,
    "joint_regularization": joint_regularization,
    # recovery
    "recovery_progress": recovery_progress,
    "legs_not_in_air": legs_not_in_air,
    "energy_efficiency": energy_efficiency,
    "forward_progress": forward_progress,
    "minimize_base_roll": minimize_base_roll,
    # upside-down standup
    "standup_height": standup_height,
    "complete_standup": complete_standup,
    "height_when_upright": height_when_upright,
}
