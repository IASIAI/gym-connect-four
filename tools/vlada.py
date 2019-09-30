import gym
from gym_connect_four import RandomPlayer, ConnectFourEnv, Player, SavedPlayer
import numpy as np


class Vlada(Player):
    """ Clone of RandomPlayer for runner.py illustration purpose """
    def __init__(self, env, name='RandomPlayer'):
        super(Vlada, self).__init__(env, name)

    def get_next_action(self, state: np.ndarray) -> int:
        for _ in range(100):
            action = np.random.randint(self.env.action_space.n)
            if self.env.is_valid_action(action):
                return action
        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')
