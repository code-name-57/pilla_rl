"""YAML configuration loader and environment factory.

Usage::

    from pilla_rl.config_loader import load_task_config, instantiate_env

    cfg = load_task_config("pilla_rl/configs/tasks/walk.yaml")
    env = instantiate_env(cfg, num_envs=4096, show_viewer=False)

Dependencies: pyyaml (``pip install pyyaml``)
"""

import importlib
from pathlib import Path

import yaml


def load_task_config(config_path: str) -> dict:
    """Load a YAML task config file and return it as a plain dict.

    Parameters
    ----------
    config_path:
        Path to the ``.yaml`` file, either absolute or relative to the
        current working directory.

    Returns
    -------
    dict
        The parsed configuration.
    """
    config_path = Path(config_path)
    with config_path.open("r") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def instantiate_env(config: dict, num_envs: int = 4096, show_viewer: bool = False):
    """Dynamically import the env class specified in *config* and instantiate it.

    The config dict must contain an ``env_class`` key with a fully-qualified
    class name, e.g. ``"pilla_rl.envs.walk_env.WalkEnv"``.

    Parameters
    ----------
    config:
        Parsed task config dict (as returned by :func:`load_task_config`).
    num_envs:
        Number of parallel simulation environments.
    show_viewer:
        Whether to open the Genesis viewer window.

    Returns
    -------
    BaseQuadrupedEnv
        An instantiated environment object.
    """
    env_class_path: str = config["env_class"]
    module_path, class_name = env_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    env_cls = getattr(module, class_name)

    return env_cls(
        num_envs=num_envs,
        env_cfg=config["env_cfg"],
        obs_cfg=config["obs_cfg"],
        reward_cfg=config["reward_cfg"],
        command_cfg=config["command_cfg"],
        show_viewer=show_viewer,
    )
