from abc import ABC, abstractmethod
from typing import Tuple

import gym
import numpy as np
from gym import spaces, logger
from keras.engine.saving import load_model
import pygame
from gym_connect_four.envs.render import render_board
from gym import error

class Player(ABC):
    """ Class used for evaluating the game """

    def __init__(self, env, name='Player'):
        self.name = name
        self.env = env

    @abstractmethod
    def get_next_action(self, state: np.ndarray) -> int:
        pass

    def learn(self, state, action, state_next, reward, done) -> None:
        pass


class RandomPlayer(Player):
    def __init__(self, env, name='RandomPlayer'):
        super(RandomPlayer, self).__init__(env, name)

    def get_next_action(self, state: np.ndarray) -> int:
        for _ in range(100):
            action = np.random.randint(self.env.action_space.n)
            if self.env.is_valid_action(action):
                return action
        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')


class SavedPlayer(Player):
    def __init__(self, env, name='SavedPlayer', model_prefix=None):
        super(SavedPlayer, self).__init__(env, name)

        if model_prefix is None:
            model_prefix = self.name

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.model = load_model(f"{model_prefix}.h5")

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        for _ in range(100):
            q_values = self.model.predict(state)
            q_values = np.array([[x if idx in self.env.available_moves() else -10 for idx, x in enumerate(q_values[0])]])
            action = np.argmax(q_values[0])
            if self.env.is_valid_action(action):
                return action

        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')


class ConnectFourEnv(gym.Env):
    """
    Description:
        ConnectFour game environment

    Observation:
        Type: Discreet(6,7)

    Actions:
        Type: Discreet(7)
        Num     Action
        x       Column in which to insert next token (0-6)

    Reward:
        Reward is 0 for every step.
        If there are no other further steps possible, Reward is 0.5 and termination will occur
        If it's a win condition, Reward will be 1 and termination will occur
        If it is an invalid move, Reward will be -1 and termination will occur

    Starting State:
        All observations are assigned a value of 0

    Episode Termination:
        No more spaces left for pieces
        4 pieces are present in a line: horizontal, vertical or diagonally
        An attempt is made to place a piece in an invalid location
    """

    metadata = {'render.modes': ['human']}

    LOSS_REWARD = -1
    DEF_REWARD = 0
    DRAW_REWARD = 0.5
    WIN_REWARD = 1

    def __init__(self, board_shape=(6, 7), window_width=512, window_height=512):
        super(ConnectFourEnv, self).__init__()

        self.board_shape = board_shape

        self.observation_space = spaces.Box(low=-1, high=1, shape=board_shape, dtype=int)
        self.action_space = spaces.Discrete(board_shape[1])

        self.current_player = 1
        self.board = np.zeros(self.board_shape, dtype=int)

        self.opponent = None
        self.player_color = None
        self.rendered_board = None
        self.screen = None
        self.window_width = window_width
        self.window_height = window_height
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, _ = self._step(action)

        if done or not self.opponent:
            return self.board, reward, done, {}

        if self.current_player != self.player_color:
            # Run step loop again for the opponent player. State will be the board, but reward is the reverse of the
            # opponent's reward
            action_opponent = self.opponent.get_next_action(self.board)
            new_state, new_reward, new_done, _ = self._step(action_opponent)
            state = new_state
            reward = self._reverse_reward(new_reward)
            done = new_done

        return self.board, reward, done, {}

    def _step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self.DEF_REWARD
        done = False

        if not self.is_valid_action(action):
            raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')

        # Check and perform action
        for index in list(reversed(range(self.board_shape[0]))):
            if self.board[index][action] == 0:
                self.board[index][action] = self.current_player
                break

        self.current_player *= -1

        # Check if board is completely filled
        if np.count_nonzero(self.board[0]) == self.board_shape[1]:
            reward = self.DRAW_REWARD
            done = True
        else:
            # Check win condition
            if self.is_win_state():
                done = True
                reward = self.WIN_REWARD

        return self.board, reward, done, {}

    def reset(self, opponent: Player = None, player_color: int = 1) -> np.ndarray:
        self.opponent = opponent
        self.player_color = player_color

        self.current_player = 1
        self.board = np.zeros(self.board_shape, dtype=int)
        
        self._update_board_render()
        if opponent and self.player_color != self.current_player:
            action_opponent = self.opponent.get_next_action(self.board)
            self._step(action_opponent)

        return self.board

    def render(self, mode: str = 'human', close: bool = False) -> None:
        if mode == 'console':
            print(np.flip(self.board, axis=0))
        elif mode == 'human':
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((round(self.window_width), round(self.window_height)))

            if close:
                pygame.quit()
            
            self._update_board_render()
            frame = self.rendered_board
            surface = pygame.surfarray.make_surface(frame)
            surface = pygame.transform.rotate(surface, 90)
            self.screen.blit(surface, (0, 0))

            pygame.display.update()
        else:
            raise error.UnsupportedMode() 

    def close(self) -> None:
        pygame.quit()

    def is_valid_action(self, action: int) -> bool:
        if self.board[0][action] == 0:
            return True

        return False

    def _update_board_render(self):
        self.rendered_board = render_board(self.board,
                                           image_width=self.window_width,
                                           image_height=self.window_height)

    def is_win_state(self) -> bool:
        # Test rows
        for i in range(self.board_shape[0]):
            for j in range(self.board_shape[1] - 3):
                value = sum(self.board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test columns on transpose array
        reversed_board = [list(i) for i in zip(*self.board)]
        for i in range(self.board_shape[1]):
            for j in range(self.board_shape[0] - 3):
                value = sum(reversed_board[i][j:j + 4])
                if abs(value) == 4:
                    return True

        # Test diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += self.board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        reversed_board = np.fliplr(self.board)
        # Test reverse diagonal
        for i in range(self.board_shape[0] - 3):
            for j in range(self.board_shape[1] - 3):
                value = 0
                for k in range(4):
                    value += reversed_board[i + k][j + k]
                    if abs(value) == 4:
                        return True

        return False

    def _reverse_reward(self, reward):
        if reward == self.LOSS_REWARD:
            return self.WIN_REWARD
        elif reward == self.DEF_REWARD:
            return self.DEF_REWARD
        elif reward == self.DRAW_REWARD:
            return self.DRAW_REWARD
        elif reward == self.WIN_REWARD:
            return self.LOSS_REWARD

        return 0

    def available_moves(self) -> frozenset:
        return frozenset((i for i in range(self.board_shape[1]) if self.is_valid_action(i)))
