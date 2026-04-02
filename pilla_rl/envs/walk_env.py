"""WalkEnv — thin subclass of BaseQuadrupedEnv for the walking task.

Uses all defaults from the base class:
* termination on pitch/roll limit or episode timeout
* reset to upright standing pose
* 45-dim observation vector
"""

from pilla_rl.envs.base_env import BaseQuadrupedEnv


class WalkEnv(BaseQuadrupedEnv):
    """Walking task environment.  All behaviour is inherited from the base class."""
