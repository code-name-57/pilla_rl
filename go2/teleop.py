import argparse
import os
import pickle
from importlib import metadata
import torch
import pygame
import time
import threading
import numpy as np

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

# Import environment from walk directory
from walk.go2_env import Go2Env


class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            print("No gamepad detected! Please connect a gamepad.")
            raise Exception("No gamepad found")
        
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Gamepad connected: {self.joystick.get_name()}")
        
        # Command limits
        self.max_lin_vel = 1.0
        self.max_ang_vel = 1.0
        
        # Command smoothing
        self.prev_lin_x = 0.0
        self.prev_lin_y = 0.0
        self.prev_ang_z = 0.0
        self.smoothing = 0.7
        
        # Cached commands (thread-safe)
        self._cached_commands = [0.0, 0.0, 0.0]
        self._reset_requested = False
        self._command_lock = threading.Lock()
        self._running = True
        
        # Start background thread for gamepad polling
        self._gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self._gamepad_thread.start()
        
    def _gamepad_loop(self):
        """Background thread that continuously polls gamepad input"""
        target_fps = 60
        dt = 1.0 / target_fps
        
        while self._running:
            start_time = time.time()
            
            try:
                pygame.event.pump()
                
                # Left stick for linear velocity (X and Y)
                lin_x = self.joystick.get_axis(1) * -self.max_lin_vel  # Forward/backward (inverted)
                lin_y = self.joystick.get_axis(0) * -self.max_lin_vel  # Left/right (inverted)
                
                # Right stick X-axis for angular velocity
                ang_z = self.joystick.get_axis(3) * -self.max_ang_vel  # Rotation (inverted)
                
                # Check for reset button (e.g., button 0 - usually 'A' or 'X')
                reset_pressed = self.joystick.get_button(0)
                
                # Apply deadzone
                deadzone = 0.15
                if abs(lin_x) < deadzone:
                    lin_x = 0.0
                if abs(lin_y) < deadzone:
                    lin_y = 0.0
                if abs(ang_z) < deadzone:
                    ang_z = 0.0
                
                # Apply smoothing
                lin_x = self.smoothing * self.prev_lin_x + (1 - self.smoothing) * lin_x
                lin_y = self.smoothing * self.prev_lin_y + (1 - self.smoothing) * lin_y
                ang_z = self.smoothing * self.prev_ang_z + (1 - self.smoothing) * ang_z
                
                # Store for next iteration
                self.prev_lin_x = lin_x
                self.prev_lin_y = lin_y
                self.prev_ang_z = ang_z
                
                # Update cached commands thread-safely
                with self._command_lock:
                    self._cached_commands = [lin_x, lin_y, ang_z]
                    if reset_pressed:
                        self._reset_requested = True
                    
            except Exception as e:
                print(f"Gamepad polling error: {e}")
                
            # Maintain target frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
    def get_commands(self):
        """Get the latest cached commands (non-blocking)"""
        with self._command_lock:
            return self._cached_commands[0], self._cached_commands[1], self._cached_commands[2]
    
    def check_reset_requested(self):
        """Check if reset was requested and clear the flag"""
        with self._command_lock:
            if self._reset_requested:
                self._reset_requested = False
                return True
            return False
    
    def stop(self):
        """Stop the gamepad polling thread"""
        self._running = False
        if self._gamepad_thread.is_alive():
            self._gamepad_thread.join(timeout=1.0)


def load_latest_checkpoint(log_dir):
    """Load the latest checkpoint from the specified log directory."""
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} does not exist")
    
    checkpoints = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {log_dir}")
    
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[1].split(".")[0]))
    checkpoint_num = int(latest_checkpoint.split("_")[1].split(".")[0])
    print(f"Loading checkpoint {checkpoint_num} from {log_dir}")
    return os.path.join(log_dir, latest_checkpoint)


def determine_robot_state(obs, dof_pos, base_quat):
    """
    Determine the robot state based on observation, DOF positions, and base orientation.
    
    Args:
        obs: Observation tensor
        dof_pos: Joint positions
        base_quat: Base quaternion [w, x, y, z]
    
    Returns:
        str: Robot state ('upside_down', 'standup')
    """
    # Convert quaternion to rotation matrix to check orientation
    # Check if robot is upside down by looking at the z-component of the up vector
    w, x, y, z = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
    
    # Calculate the z-component of the world up vector in body frame
    # This tells us how much the robot's up direction aligns with world up
    up_z = 2 * (w * z + x * y)
    
    # Check base height (assuming it's in the observation)
    base_height = obs[0, 2] if obs.shape[1] > 2 else 0.0  # Adjust index based on your obs structure
    
    # State determination logic
    if up_z < -0.5:  # Robot is significantly upside down
        return "upside_down"
    else:  # Robot is right-side up and ready to stand up
        return "standup"


def create_upside_down_env(env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=True):
    """Create an environment with the robot initialized upside down."""
    # Modify environment config to start upside down
    env_cfg_modified = env_cfg.copy()
    env_cfg_modified["base_init_pos"] = [0.0, 0.0, 0.5]  # Start slightly elevated
    env_cfg_modified["base_init_rot"] = [0.0, 0.0, 1.0, 0.0]  # Upside down (180° around y-axis)
    
    return Go2Env(
        num_envs=1,
        env_cfg=env_cfg_modified,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_viewer", action="store_true", default=True, help="Show the viewer")
    args = parser.parse_args()

    gs.init()

    # Define policy directories (removed walk)
    policy_dirs = {
        "upside_down_recovery": "upside_down_recovery/logs",
        "standup_copilot": "standup_copilot/logs"
    }
    
    # Find the actual log directories (they might have experiment names)
    actual_dirs = {}
    for policy_name, base_dir in policy_dirs.items():
        if os.path.exists(base_dir):
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if subdirs:
                # Use the first (or most recent) subdirectory
                actual_dirs[policy_name] = os.path.join(base_dir, subdirs[0])
            else:
                print(f"Warning: No subdirectories found in {base_dir}")
        else:
            print(f"Warning: Directory {base_dir} does not exist")
    
    # Load policies and environments
    policies = {}
    envs = {}
    
    print("Loading policies...")
    for policy_name, log_dir in actual_dirs.items():
        try:
            print(f"Loading {policy_name} from {log_dir}")
            
            # Load configuration
            cfg_path = os.path.join(log_dir, "cfgs.pkl")
            if not os.path.exists(cfg_path):
                print(f"Warning: Config file not found at {cfg_path}")
                continue
                
            env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(cfg_path, "rb"))
            reward_cfg["reward_scales"] = {}
            
            # Create environment for this policy
            if policy_name == "upside_down_recovery":
                env = create_upside_down_env(env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False)
            else:
                env_cfg["base_init_pos"] = [0.0, 0.0, 0.5]
                env = Go2Env(
                    num_envs=1,
                    env_cfg=env_cfg,
                    obs_cfg=obs_cfg,
                    reward_cfg=reward_cfg,
                    command_cfg=command_cfg,
                    show_viewer=False,
                )
            
            # Load policy
            runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
            resume_path = load_latest_checkpoint(log_dir)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=gs.device)
            
            policies[policy_name] = policy
            envs[policy_name] = env
            
            print(f"Successfully loaded {policy_name}")
            
        except Exception as e:
            print(f"Failed to load {policy_name}: {e}")
    
    if not policies:
        print("Error: No policies were loaded successfully!")
        return
    
    # Use the standup environment as the main environment with viewer
    main_env_cfg, main_obs_cfg, main_reward_cfg, main_command_cfg, _ = pickle.load(
        open(os.path.join(actual_dirs["standup_copilot"], "cfgs.pkl"), "rb")
    )
    main_reward_cfg["reward_scales"] = {}
    
    # Create main environment that starts upside down
    env = create_upside_down_env(main_env_cfg, main_obs_cfg, main_reward_cfg, main_command_cfg, show_viewer=args.show_viewer)
    
    # Initialize gamepad controller
    try:
        gamepad = GamepadController()
        print("\nGamepad controls:")
        print("- Left stick: Forward/backward and left/right movement")
        print("- Right stick (X-axis): Rotation")
        print("- Button A/X: Reset simulation (robot will start upside down)")
        print("- Press Ctrl+C to exit")
    except Exception as e:
        print(f"Gamepad initialization failed: {e}")
        print("Continuing without gamepad (robot will use zero commands)...")
        gamepad = None

    # Reset environment to start upside down
    obs, _ = env.reset()
    
    print("\nRobot initialized upside down. Automatic recovery sequence will begin...")
    print("Available policies:", list(policies.keys()))
    
    current_policy_name = "upside_down_recovery"
    current_policy = policies.get(current_policy_name, list(policies.values())[0])
    
    step_count = 0
    last_state_change = 0
    state_change_cooldown = 50  # Minimum steps before allowing state change
    
    with torch.no_grad():
        while True:
            try:
                # Check for reset request
                if gamepad and gamepad.check_reset_requested():
                    print("Reset requested! Reinitializing robot upside down...")
                    obs, _ = env.reset()
                    current_policy_name = "upside_down_recovery"
                    current_policy = policies.get(current_policy_name, list(policies.values())[0])
                    step_count = 0
                    last_state_change = 0
                    continue
                
                # Get gamepad commands
                if gamepad:
                    lin_x, lin_y, ang_z = gamepad.get_commands()
                else:
                    lin_x = 0.0
                    lin_y = 0.0
                    ang_z = 0.0
                
                # Get robot state information
                # Note: You may need to adjust these indices based on your observation structure
                base_pos = obs[0, :3] if obs.shape[1] >= 3 else torch.zeros(3)
                base_quat = obs[0, 3:7] if obs.shape[1] >= 7 else torch.tensor([1.0, 0.0, 0.0, 0.0])
                dof_pos = obs[0, 7:19] if obs.shape[1] >= 19 else torch.zeros(12)  # Assuming 12 DOF
                
                # Determine robot state
                robot_state = determine_robot_state(obs, dof_pos, base_quat)
                
                # Switch policy if needed (with cooldown to prevent rapid switching)
                if step_count - last_state_change > state_change_cooldown:
                    if robot_state != current_policy_name and robot_state in policies:
                        print(f"State changed: {current_policy_name} -> {robot_state}")
                        current_policy_name = robot_state
                        current_policy = policies[robot_state]
                        last_state_change = step_count
                
                # Get actions from current policy
                actions = current_policy(obs)
                
                # Set commands
                env.commands = torch.tensor([[lin_x, lin_y, ang_z]], dtype=torch.float).to(gs.device)
                
                # Step environment
                obs, rews, dones, infos = env.step(actions)
                
                # Print status every 100 steps
                if step_count % 100 == 0:
                    print(f"Step {step_count}: State={robot_state}, Policy={current_policy_name}, "
                          f"Base height={base_pos[2]:.3f}, Commands=[{lin_x:.2f}, {lin_y:.2f}, {ang_z:.2f}]")
                
                step_count += 1
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    if gamepad:
        gamepad.stop()
        pygame.quit()
    
    print("Teleop session ended.")


if __name__ == "__main__":
    main()
