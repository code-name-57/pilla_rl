import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2Env:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
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

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
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

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
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
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
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

        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )
        self._resample_commands(envs_idx)

        # check termination and reset (very lenient for upside-down recovery learning)
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        # Don't terminate on orientation - allow robot to be upside-down and learn to recover
        # Only terminate if robot gets stuck in very bad positions or falls off the world
        # self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        # self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]
        
        # Only reset if base gets extremely low (completely stuck/fallen through ground)
        self.reset_buf |= self.base_pos[:, 2] < 0.05

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                self.base_euler * torch.pi / 180,  # 3 - Add base orientation (roll, pitch, yaw)
            ],
            axis=-1,
        )

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

        dofs_lower_limits, dofs_upper_limits = self.robot.get_dofs_limit(self.motors_dof_idx)

        # reset dofs to random positions based on joint limits
        self.dof_pos[envs_idx] = gs_rand_float(
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

        # reset base to upside-down positions for recovery training
        # Focus on upside-down and challenging orientations that require recovery
        
        # Randomize base position - robot can be on its back on the ground
        self.base_pos[envs_idx] = self.base_init_pos + gs_rand_float(
            lower=torch.tensor([-0.2, -0.2, 0.15], device=gs.device),  # Keep some height for recovery
            upper=torch.tensor([0.2, 0.2, 0.5], device=gs.device),
            shape=(len(envs_idx), 3),
            device=gs.device,
        )
        
        # Generate upside-down and challenging orientations
        # 70% chance for upside-down positions (roll ~180°, various pitch angles)
        # 30% chance for other challenging positions (high roll/pitch but not fully upside-down)
        prob_upside_down = torch.rand(len(envs_idx), device=gs.device)
        
        random_euler = torch.zeros((len(envs_idx), 3), device=gs.device)
        
        # Upside-down positions (roll around 180°)
        upside_down_mask = prob_upside_down < 0.7
        upside_down_idx = upside_down_mask.nonzero(as_tuple=False).flatten()
        if len(upside_down_idx) > 0:
            # Roll: 150° to 210° (around upside-down)
            random_euler[upside_down_idx, 0] = gs_rand_float(
                lower=torch.tensor(150.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(210.0, device=gs.device) * torch.pi / 180,
                shape=(len(upside_down_idx),),
                device=gs.device,
            )
            # Pitch: -60° to 60° for variety
            random_euler[upside_down_idx, 1] = gs_rand_float(
                lower=torch.tensor(-60.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(60.0, device=gs.device) * torch.pi / 180,
                shape=(len(upside_down_idx),),
                device=gs.device,
            )
            # Yaw: full range
            random_euler[upside_down_idx, 2] = gs_rand_float(
                lower=torch.tensor(-180.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(180.0, device=gs.device) * torch.pi / 180,
                shape=(len(upside_down_idx),),
                device=gs.device,
            )
        
        # Other challenging positions
        other_mask = ~upside_down_mask
        other_idx = other_mask.nonzero(as_tuple=False).flatten()
        if len(other_idx) > 0:
            # Roll: 60° to 120° or -120° to -60° (on sides)
            side_roll = torch.rand(len(other_idx), device=gs.device)
            random_euler[other_idx, 0] = torch.where(
                side_roll < 0.5,
                gs_rand_float(60.0, 120.0, (len(other_idx),), gs.device) * torch.pi / 180,
                gs_rand_float(-120.0, -60.0, (len(other_idx),), gs.device) * torch.pi / 180
            )
            # Pitch: -90° to 90°
            random_euler[other_idx, 1] = gs_rand_float(
                lower=torch.tensor(-90.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(90.0, device=gs.device) * torch.pi / 180,
                shape=(len(other_idx),),
                device=gs.device,
            )
            # Yaw: full range
            random_euler[other_idx, 2] = gs_rand_float(
                lower=torch.tensor(-180.0, device=gs.device) * torch.pi / 180,
                upper=torch.tensor(180.0, device=gs.device) * torch.pi / 180,
                shape=(len(other_idx),),
                device=gs.device,
            )
        
        # Convert random euler angles to quaternions
        from genesis.utils.geom import xyz_to_quat
        random_quat = xyz_to_quat(random_euler, rpy=True, degrees=False)
        self.base_quat[envs_idx] = random_quat
        
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
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

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
    
    def _reward_upright_orientation(self):
        # Strong reward for transitioning from upside-down to upright
        # Calculate how "upright" the robot is (roll and pitch close to 0)
        roll_rad = self.base_euler[:, 0] * torch.pi / 180
        pitch_rad = self.base_euler[:, 1] * torch.pi / 180
        
        # Reward based on how close to upright (0 roll, 0 pitch)
        upright_score = torch.cos(roll_rad) * torch.cos(pitch_rad)
        # Convert to positive reward (1 when upright, -1 when upside-down)
        return upright_score
    
    def _reward_stability(self):
        # Reward for controlled movement (not too chaotic)
        ang_vel_penalty = torch.sum(torch.square(self.base_ang_vel), dim=1)
        return torch.exp(-ang_vel_penalty / 1.0)
    
    def _reward_recovery_progress(self):
        # Main reward for upside-down recovery progress
        roll_rad = self.base_euler[:, 0] * torch.pi / 180
        pitch_rad = self.base_euler[:, 1] * torch.pi / 180
        
        # Reward getting closer to upright orientation
        # Use cosine for smooth gradients from upside-down positions
        roll_progress = (torch.cos(roll_rad) + 1.0) / 2.0  # 0 when upside-down, 1 when upright
        pitch_progress = (torch.cos(pitch_rad) + 1.0) / 2.0  # 0 when upside-down, 1 when upright
        
        # Combine roll and pitch progress
        orientation_progress = roll_progress * pitch_progress
        
        # Add height component - reward for maintaining reasonable height during recovery
        height_factor = torch.exp(-torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"]) / 0.2)
        
        return orientation_progress * (0.7 + 0.3 * height_factor)
    
    def _reward_legs_not_in_air(self):
        # Encourage legs to be grounded or at least not flailing
        # Penalize excessive joint velocities which might indicate uncontrolled flailing
        joint_vel_penalty = torch.sum(torch.square(self.dof_vel), dim=1)
        return torch.exp(-joint_vel_penalty / 20.0)
    
    def _reward_energy_efficiency(self):
        # Encourage efficient movements - penalize excessive joint torques
        action_magnitude = torch.sum(torch.square(self.actions), dim=1)
        return torch.exp(-action_magnitude / 5.0)
    
    def _reward_forward_progress(self):
        # Small reward for any forward movement during recovery (helps with getting up)
        # Only reward if moving in reasonable direction and not upside-down
        roll_rad = self.base_euler[:, 0] * torch.pi / 180
        is_reasonably_upright = torch.abs(roll_rad) < (torch.pi / 2)  # Less than 90 degrees roll
        
        forward_vel = self.base_lin_vel[:, 0]  # X-axis velocity
        forward_reward = torch.clamp(forward_vel, 0.0, 1.0)  # Only reward forward, cap at 1 m/s
        
        return torch.where(is_reasonably_upright, forward_reward, torch.zeros_like(forward_reward))
    
    def _reward_minimize_base_roll(self):
        # Specific reward for minimizing roll angle (key for upside-down recovery)
        roll_rad = torch.abs(self.base_euler[:, 0] * torch.pi / 180)
        # Smooth reward that provides gradients even when upside-down
        return torch.exp(-roll_rad / (torch.pi / 4))  # Decay over 45 degrees
