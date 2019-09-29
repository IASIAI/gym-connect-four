from gym.envs.registration import register
from .envs.connect_four_env import ConnectFourEnv, Player, RandomPlayer, SavedPlayer

register(
    id='ConnectFour-v0',
    entry_point='gym_connect_four.envs:ConnectFourEnv',
)
