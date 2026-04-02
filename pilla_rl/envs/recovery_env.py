"""RecoveryEnv — thin subclass of BaseQuadrupedEnv for upside-down recovery /
standup tasks.

Differences from the base class:
* Very lenient termination (only base_pos z < 0.05 or episode timeout)
* Upside-down biased reset (70% upside-down ~180° roll, 30% on sides)
* 48-dim observation vector (adds base_euler * pi/180)
"""

import torch
import genesis as gs
from genesis.utils.geom import xyz_to_quat

from pilla_rl.envs.base_env import BaseQuadrupedEnv, _gs_rand_float


class RecoveryEnv(BaseQuadrupedEnv):
    """Recovery / upside-down standup task environment."""

    # ------------------------------------------------------------------
    # Termination: only z < 0.05 or episode timeout (no orientation limit)
    # ------------------------------------------------------------------

    def _check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.base_pos[:, 2] < 0.05

    # ------------------------------------------------------------------
    # Reset: upside-down biased orientations (70/30 split)
    # ------------------------------------------------------------------

    def _reset_robot_pose(self, envs_idx):
        dofs_lower_limits, dofs_upper_limits = self.robot.get_dofs_limit(self.motors_dof_idx)

        # Random DOF positions within joint limits
        self.dof_pos[envs_idx] = _gs_rand_float(
            lower=dofs_lower_limits,
            upper=dofs_upper_limits,
            shape=(len(envs_idx), self.num_actions),
            device=gs.device,
        )
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Base position — keep some height for recovery
        self.base_pos[envs_idx] = self.base_init_pos + _gs_rand_float(
            lower=torch.tensor([-0.2, -0.2, 0.15], device=gs.device),
            upper=torch.tensor([0.2, 0.2, 0.5], device=gs.device),
            shape=(len(envs_idx), 3),
            device=gs.device,
        )

        # 70% upside-down, 30% on sides
        prob_upside_down = torch.rand(len(envs_idx), device=gs.device)
        random_euler = torch.zeros((len(envs_idx), 3), device=gs.device)

        upside_down_mask = prob_upside_down < 0.7
        upside_down_idx = upside_down_mask.nonzero(as_tuple=False).flatten()
        if len(upside_down_idx) > 0:
            # Roll: 150° to 210° (around upside-down)
            random_euler[upside_down_idx, 0] = _gs_rand_float(
                lower=torch.tensor(150.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(210.0, device=gs.device) * torch.pi / 180,
                shape=(len(upside_down_idx),),
                device=gs.device,
            )
            # Pitch: -60° to 60° for variety
            random_euler[upside_down_idx, 1] = _gs_rand_float(
                lower=torch.tensor(-60.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(60.0, device=gs.device) * torch.pi / 180,
                shape=(len(upside_down_idx),),
                device=gs.device,
            )
            # Yaw: full range
            random_euler[upside_down_idx, 2] = _gs_rand_float(
                lower=torch.tensor(-180.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(180.0, device=gs.device) * torch.pi / 180,
                shape=(len(upside_down_idx),),
                device=gs.device,
            )

        other_mask = ~upside_down_mask
        other_idx = other_mask.nonzero(as_tuple=False).flatten()
        if len(other_idx) > 0:
            # Roll: 60° to 120° or -120° to -60° (on sides)
            side_roll = torch.rand(len(other_idx), device=gs.device)
            random_euler[other_idx, 0] = torch.where(
                side_roll < 0.5,
                _gs_rand_float(60.0, 120.0, (len(other_idx),), gs.device) * torch.pi / 180,
                _gs_rand_float(-120.0, -60.0, (len(other_idx),), gs.device) * torch.pi / 180,
            )
            # Pitch: -90° to 90°
            random_euler[other_idx, 1] = _gs_rand_float(
                lower=torch.tensor(-90.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(90.0, device=gs.device) * torch.pi / 180,
                shape=(len(other_idx),),
                device=gs.device,
            )
            # Yaw: full range
            random_euler[other_idx, 2] = _gs_rand_float(
                lower=torch.tensor(-180.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(180.0, device=gs.device) * torch.pi / 180,
                shape=(len(other_idx),),
                device=gs.device,
            )

        random_quat = xyz_to_quat(random_euler, rpy=True, degrees=False)
        self.base_quat[envs_idx] = random_quat

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

    # ------------------------------------------------------------------
    # Observations: 48-dim (base + euler angles in radians)
    # ------------------------------------------------------------------

    def _compute_observations(self):
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],                           # 3
                self.projected_gravity,                                                    # 3
                self.commands * self.commands_scale,                                       # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],       # 12
                self.dof_vel * self.obs_scales["dof_vel"],                                 # 12
                self.actions,                                                              # 12
                self.base_euler * torch.pi / 180,                                          # 3
            ],
            axis=-1,
        )
