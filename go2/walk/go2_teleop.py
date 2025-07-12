import argparse
import os
import pickle
from importlib import metadata
import torch
import pygame
import time
import threading

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
        
        # Reduce command limits for stability
        self.max_lin_vel = 1.0  # Reduced from 2.0
        self.max_ang_vel = 1.0  # Reduced from 2.0
        
        # Add command smoothing
        self.prev_lin_x = 0.0
        self.prev_lin_y = 0.0
        self.prev_ang_z = 0.0
        self.smoothing = 0.7  # Smoothing factor
        
        # Cached commands (thread-safe)
        self._cached_commands = [0.0, 0.0, 0.0]
        self._command_lock = threading.Lock()
        self._running = True
        
        # Start background thread for gamepad polling
        self._gamepad_thread = threading.Thread(target=self._gamepad_loop, daemon=True)
        self._gamepad_thread.start()
        
    def _gamepad_loop(self):
        """Background thread that continuously polls gamepad input"""
        target_fps = 60  # Poll gamepad at 60Hz
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
                
                # Apply deadzone
                deadzone = 0.15  # Increased deadzone
                if abs(lin_x) < deadzone:
                    lin_x = 0.0
                if abs(lin_y) < deadzone:
                    lin_y = 0.0
                if abs(ang_z) < deadzone:
                    ang_z = 0.0
                
                # Apply smoothing to prevent sudden changes
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
    
    def stop(self):
        """Stop the gamepad polling thread"""
        self._running = False
        if self._gamepad_thread.is_alive():
            self._gamepad_thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}
    env_cfg["base_init_pos"] = [0.0, 0.0, 0.5]
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

    # Initialize gamepad controller
    try:
        gamepad = GamepadController()
        print("Gamepad controls:")
        print("- Left stick: Forward/backward and left/right movement")
        print("- Right stick (X-axis): Rotation")
        print("- Press Ctrl+C to exit")
    except Exception as e:
        print(f"Gamepad initialization failed: {e}")
        print("Falling back to static commands...")
        gamepad = None

    obs, _ = env.reset()
    
    # Initialize with zero commands and let robot stabilize
    print("Letting robot stabilize for 2 seconds...")
    for _ in range(100):  # Assuming 50Hz, this is 2 seconds
        actions = policy(obs)
        env.commands = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float).to("cuda:0").repeat(1, 1)
        obs, rews, dones, infos = env.step(actions)
        time.sleep(0.02)
    
    print("Robot stabilized. Gamepad control active.")
    
    with torch.no_grad():
        while True:
            try:                
                if gamepad:
                    lin_x, lin_y, ang_z = gamepad.get_commands()
                else:
                    lin_x = 0.0
                    lin_y = 0.0
                    ang_z = 0.0

                actions = policy(obs)
                env.commands = torch.tensor([[lin_x, lin_y, ang_z]], dtype=torch.float).to("cuda:0").repeat(1, 1)
                obs, rews, dones, infos = env.step(actions)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    
    if gamepad:
        gamepad.stop()
        pygame.quit()


if __name__ == "__main__":
    main()