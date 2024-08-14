"""Policy package."""
# isort:skip_file

from LCRL.policy.base import BasePolicy
from LCRL.policy.random import RandomPolicy
from LCRL.policy.modelfree.ddpg import DDPGPolicy
from LCRL.policy.modelfree.sac import SACPolicy


# the belows are new
from LCRL.policy.modelfree.ddpg_reach_avoid_game_new import reach_avoid_game_DDPGPolicy
from LCRL.policy.modelfree.ddpg_reach_avoid_game_classical import reach_avoid_game_DDPGPolicy_annealing
from LCRL.policy.modelfree.sac_reach_avoid_game_new import reach_avoid_game_SACPolicy
from LCRL.policy.modelfree.sac_reach_avoid_game_classical import reach_avoid_game_SACPolicy_annealing

__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "DDPGPolicy",
    "SACPolicy",
    "reach_avoid_game_DDPGPolicy_annealing", # arXiv:2112.12288, implemented using DDPG
    "reach_avoid_game_DDPGPolicy", # Our new method, implemented using DDPG
    "reach_avoid_game_SACPolicy_annealing", # arXiv:2112.12288, implemented using SAC
    "reach_avoid_game_SACPolicy", # Our new method, implemented using SAC
]

