"""StandupEnv — thin subclass of BaseQuadrupedEnv for the stand-up task.

Differences from the base class:
* Lenient termination (no pitch/roll limit; only episode timeout)
* Random pose reset (random euler angles up to ±60°; random DOF positions)
* 48-dim observation vector (adds base_euler * pi/180)
"""

import torch
import genesis as gs
from genesis.utils.geom import xyz_to_quat

from pilla_rl.envs.base_env import BaseQuadrupedEnv, _gs_rand_float


class StandupEnv(BaseQuadrupedEnv):
    """Stand-up task environment with randomised starting pose."""

    # ------------------------------------------------------------------
    # Termination: only episode length (no orientation limit)
    # ------------------------------------------------------------------

    def _check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length

    # ------------------------------------------------------------------
    # Reset: random euler angles and random DOF positions
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

        # Random base position with some variation
        self.base_pos[envs_idx] = self.base_init_pos + _gs_rand_float(
            lower=torch.tensor([-0.3, -0.3, 0.05], device=gs.device),
            upper=torch.tensor([0.3, 0.3, 0.6], device=gs.device),
            shape=(len(envs_idx), 3),
            device=gs.device,
        )

        # Random euler angles (curriculum: max ±60° for roll/pitch)
        max_roll_pitch = torch.tensor(60.0, device=gs.device)
        random_euler = _gs_rand_float(
            lower=torch.tensor([-max_roll_pitch, -max_roll_pitch, -180.0], device=gs.device) * torch.pi / 180,
            upper=torch.tensor([max_roll_pitch, max_roll_pitch, 180.0], device=gs.device) * torch.pi / 180,
            shape=(len(envs_idx), 3),
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
