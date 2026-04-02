"""Base quadruped environment shared by all task-specific subclasses.

All the logic that was previously duplicated across every per-task ``go2_env.py``
lives here.  Task-specific differences are handled by overriding the three
protected methods:

* ``_check_termination()``  — when to reset an episode
* ``_reset_robot_pose(envs_idx)`` — initial body / joint state on reset
* ``_compute_observations()`` — what the agent sees

Reward functions are **not** methods on this class.  Instead they are looked
up from ``pilla_rl.rewards.REWARD_REGISTRY`` and their arguments are
auto-resolved from the environment's attributes via ``inspect.signature()``.
"""

import inspect
import math

import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat

from pilla_rl.rewards.reward_functions import REWARD_REGISTRY


def _gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class BaseQuadrupedEnv:
    """Shared quadruped environment for all Go2 tasks."""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # ------------------------------------------------------------------
        # Scene
        # ------------------------------------------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        self.scene.build(n_envs=num_envs)

        # ------------------------------------------------------------------
        # PD control
        # ------------------------------------------------------------------
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # ------------------------------------------------------------------
        # Buffers
        # ------------------------------------------------------------------
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.base_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()
        self.extras["observations"] = dict()

        # ------------------------------------------------------------------
        # Rewards (uses REWARD_REGISTRY instead of per-class methods)
        # ------------------------------------------------------------------
        self._setup_rewards()

    # ------------------------------------------------------------------
    # Reward wiring
    # ------------------------------------------------------------------

    def _setup_rewards(self):
        """Register reward functions from REWARD_REGISTRY and scale by dt."""
        self.reward_functions = {}
        self.episode_sums = {}
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            if name not in REWARD_REGISTRY:
                raise ValueError(
                    f"Reward '{name}' is not in REWARD_REGISTRY. "
                    f"Available rewards: {sorted(REWARD_REGISTRY.keys())}"
                )
            self.reward_functions[name] = REWARD_REGISTRY[name]
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _resolve_reward_args(self, fn):
        """Map function parameter names to current environment state tensors."""
        param_map = {
            "base_lin_vel": lambda: self.base_lin_vel,
            "base_ang_vel": lambda: self.base_ang_vel,
            "base_pos": lambda: self.base_pos,
            "base_euler": lambda: self.base_euler,
            "commands": lambda: self.commands,
            "dof_pos": lambda: self.dof_pos,
            "dof_vel": lambda: self.dof_vel,
            "actions": lambda: self.actions,
            "last_actions": lambda: self.last_actions,
            "default_dof_pos": lambda: self.default_dof_pos,
            "tracking_sigma": lambda: self.reward_cfg["tracking_sigma"],
            "target_height": lambda: self.reward_cfg["base_height_target"],
            "base_height_target": lambda: self.reward_cfg["base_height_target"],
        }
        sig = inspect.signature(fn)
        kwargs = {}
        for param_name in sig.parameters:
            if param_name not in param_map:
                raise ValueError(
                    f"Reward function '{fn.__name__}' has unknown parameter '{param_name}'. "
                    f"Supported parameters: {sorted(param_map.keys())}"
                )
            kwargs[param_name] = param_map[param_name]()
        return kwargs

    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, fn in self.reward_functions.items():
            kwargs = self._resolve_reward_args(fn)
            rew = fn(**kwargs) * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    # ------------------------------------------------------------------
    # Overridable per-task methods
    # ------------------------------------------------------------------

    def _check_termination(self):
        """Default termination: episode length or pitch/roll limit."""
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

    def _reset_robot_pose(self, envs_idx):
        """Default reset: upright standing pose."""
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

    def _compute_observations(self):
        """Default 45-dim observation vector."""
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],           # 3
                self.projected_gravity,                                    # 3
                self.commands * self.commands_scale,                       # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],                 # 12
                self.actions,                                              # 12
            ],
            axis=-1,
        )

    # ------------------------------------------------------------------
    # Command resampling
    # ------------------------------------------------------------------

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = _gs_rand_float(
            *self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 1] = _gs_rand_float(
            *self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device
        )
        self.commands[envs_idx, 2] = _gs_rand_float(
            *self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device
        )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # Update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # Resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # Termination check
        self._check_termination()
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Rewards and observations
        self._compute_rewards()
        self._compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self._reset_robot_pose(envs_idx)

        # Reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None
