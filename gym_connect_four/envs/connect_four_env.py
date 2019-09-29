from abc import ABC, abstractmethod
from threading import Lock, Event
from typing import Tuple

import gym
import numpy as np
from gym import spaces


class OpponentSyncEnv(gym.Env):
    """
    A generic base Environment class for games with 1 opponent and syncronized moves
    """
    PLAYERS = {-1, 1}

    def __init__(self) -> None:
        super().__init__()
        self._player_events = [Event(), Event()]
        self._current_player = 1
        self._player_change_lock = Lock()

    @abstractmethod
    def is_valid_action(self, action: int) -> bool:
        ...

    def _get_player_event(self, current_player: int):
        return self._player_events[(current_player + 1) // 2]

    def _next_player(self):
        with self._player_change_lock:
            self._get_player_event(self._current_player).clear()
            self._current_player *= -1
            self._get_player_event(self._current_player).set()
        return self._current_player

    def step_player(self, action: int, player: int):
        assert player in self.PLAYERS

        self._get_player_event(player).wait()
        state, reward, done, info = self.step(action)
        self._next_player()
        return state, reward, done, info


class Player(ABC):
    """ Class used for evaluating the game """

    def __init__(self, env: OpponentSyncEnv, id: int, name='Player'):
        self.name = name
        self.env = env
        self.id = id

    @abstractmethod
    def train(self, plays: int):
        ...

    def step(self, action):
        return self.env.step_player(action, self.id)

    def learn(self, state, action, reward, done) -> None:
        pass


class RandomPlayer(Player):
    def __init__(self, env, id: int, name='RandomPlayer'):
        super().__init__(env, id, name)

    def train(self, plays: int):
        for _ in range(plays):
            while True:
                action = np.random.randint(self.env.action_space.n)
                if self.env.is_valid_action(action):
                    state, reward, done, info = self.step(action)
                    if done:
                        break

    def get_next_action(self, state: np.ndarray) -> int:
        for _ in range(100):
            action = np.random.randint(self.env.action_space.n)
            if self.env.is_valid_action(action):
                return action
        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')


class ConnectFourEnv(OpponentSyncEnv):
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

    def __init__(self, board_shape=(6, 7)):
        super(ConnectFourEnv, self).__init__()

        self.board_shape = board_shape

        self.observation_space = spaces.Box(low=-1, high=1, shape=board_shape, dtype=int)
        self.action_space = spaces.Discrete(board_shape[1])

        self.board = np.zeros(self.board_shape, dtype=int)

        self.opponent = None
        self.player_color = None

    # def next_player(self, currrent_player: int) -> int:
    #     return (currrent_player + 1) % 2
    #
    # def play(self, player1: Player, player2: Player) -> int:
    #     """
    #     :return: -1 if wins player1, 1 if wins player2, 0 if it's a draw
    #     """
    #     players = [player1, player2]
    #     # ToDo: implement game loop from app.py
    #     return 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, _ = self._step(action)

        if done or not self.opponent:
            return self.board, reward, done, {}

        if self._current_player != self.player_color:
            # Run step loop again for the opponent player. State will be the board, but reward is the reverse of the
            # opponent's reward
            action_opponent = self.opponent.get_next_action(self.board)
            new_state, new_reward, new_done, _ = self._step(action_opponent)
            state = new_state
            reward = self._reverse_reward(new_reward)
            done = new_done

        return self.board, reward, done, {}

    def _step_player(self, action: int, player: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self.DEF_REWARD
        done = False

        if not self.is_valid_action(action):
            print("Invalid action, column is already full")
            return self.board, self.LOSS_REWARD, True, {}
        # Check and perform action
        for index in list(reversed(range(self.board_shape[0]))):
            if self.board[index][action] == 0:
                self.board[index][action] = player
                break

        self._next_player()

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

    def _step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        reward = self.DEF_REWARD
        done = False

        if not self.is_valid_action(action):
            print("Invalid action, column is already full")
            return self.board, self.LOSS_REWARD, True, {}

        # Check and perform action
        for index in list(reversed(range(self.board_shape[0]))):
            if self.board[index][action] == 0:
                self.board[index][action] = self._current_player
                break

        self._current_player *= -1

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

        self._current_player = 1
        self.board = np.zeros(self.board_shape, dtype=int)

        if opponent and self.player_color != self._current_player:
            action_opponent = self.opponent.get_next_action(self.board)
            self._step(action_opponent)

        return self.board

    def render(self, mode: str = 'human', close: bool = False) -> None:
        pass

    def close(self) -> None:
        pass

    def is_valid_action(self, action: int) -> bool:
        if self.board[0][action] == 0:
            return True

        return False

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
