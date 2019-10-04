import random
from typing import Iterable
from unittest import TestCase

import gym
import numpy as np

from envs.connect_four_env import ResultType, Player
from gym_connect_four import ConnectFourEnv, RandomPlayer

BOARD_VALIDATION = np.array([[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0],
                    [0, 0, -1, 1, 0, -1, 0],
                    [0, 0, 1, 1, 0, 1, 1],
                    [-1, -1, -1, -1, 0, -1, -1],
                    [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_ROW = np.array([[0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, -1, 0, 0, 0],
                 [0, 0, -1, 1, 0, -1, 0],
                 [0, 0, 1, 1, 0, 1, 1],
                 [-1, -1, -1, -1, 0, -1, -1],
                 [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_COLUMN = np.array([[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, -1, 1, 0, -1, 0],
                    [0, 0, 1, 1, 0, 1, 1],
                    [-1, 0, -1, -1, 0, -1, -1],
                    [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_DIAGONAL = np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, -1, 1, 0, -1, 0],
                      [0, 0, 1, 1, 1, 1, 1],
                      [-1, 1, -1, -1, 0, 1, -1],
                      [1, 1, -1, 1, -1, 1, 1]])

BOARD_WIN_BDIAGONAL = np.array([[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0],
                       [0, 0, -1, 1, 0, -1, 0],
                       [0, 0, 1, 1, 0, 1, 1],
                       [-1, 1, -1, -1, 0, -1, -1],
                       [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_0123 = np.array([[0, 0, 0, 0, -1, 1, -1],
                        [0, 0, 0, 1, 1, -1, 1],
                        [0, 0, -1, 1, 1, -1, -1],
                        [0, 0, 1, 1, 1, 1, 1],
                        [-1, 1, -1, -1, -1, -1, -1],
                        [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_2 = np.array([[1, 1, 0, -1, -1, 1, -1],
                     [1, 1, -1, 1, 1, -1, 1],
                     [1, 1, -1, 1, 1, -1, -1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [-1, 1, -1, -1, -1, -1, -1],
                     [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_6 = np.array([[1, 1, 1, 1, -1, 1, 0],
                     [1, 1, -1, 1, 1, -1, 1],
                     [1, 1, -1, 1, 1, -1, -1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [-1, 1, -1, -1, -1, -1, -1],
                     [1, 1, -1, 1, -1, 1, 1]])

BOARD_AVAILABLE_NONE = np.array([[1, 1, 1, 1, -1, 1, 1],
                        [1, 1, -1, 1, 1, -1, 1],
                        [1, 1, -1, 1, 1, -1, -1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [-1, 1, -1, -1, -1, -1, -1],
                        [1, 1, -1, 1, -1, 1, 1]])


class DeterministicPlayer(Player):
    def __init__(self, env: 'ConnectFourEnv', moves: Iterable[int], name='DeterministicPlayer'):
        super().__init__(env, name)
        self._moves = moves
        self.reset()

    def reset(self):
        self._moves_itr = iter(self._moves)
        self.action_log = []
        self.reward_log = []
        self.done_log = []
        self.states = []
        self.l_states = []
        self.l_new_states = []

    def get_next_action(self, state: np.ndarray) -> int:
        self.states.append(state)
        next_move = next(self._moves_itr)
        valid_moves = self.env.available_moves()
        while next_move not in valid_moves:
            next_move += 1
            next_move %= self.env.action_space.n
        return next_move

    def learn(self, state, action, state_next, reward, done) -> None:
        self.action_log.append(action)
        self.reward_log.append(reward)
        self.done_log.append(done)
        self.l_states.append(state)
        self.l_new_states.append(state_next)


class TestConnectFourEnv(TestCase):

    def setUp(self) -> None:
        self.env = gym.make('ConnectFour-v0')

    def test_is_valid_action(self):
        self.env = self.env
        self.env.reset(BOARD_VALIDATION)
        self.assertTrue(self.env.is_valid_action(0))
        self.assertFalse(self.env.is_valid_action(3))

    def test_is_win_state(self):
        self.env = self.env
        self.env.reset(BOARD_WIN_ROW)
        self.assertTrue(self.env.is_win_state())

        self.env.reset(BOARD_WIN_COLUMN)
        self.assertTrue(self.env.is_win_state())

        self.env.reset(BOARD_WIN_DIAGONAL)
        self.assertTrue(self.env.is_win_state())

        self.env.reset(BOARD_WIN_BDIAGONAL)
        self.assertTrue(self.env.is_win_state())

    def test_available_moves(self):
        self.env = self.env
        self.env.reset(BOARD_AVAILABLE_0123)
        self.assertEqual(set(self.env.available_moves()), {0, 1, 2, 3})

        self.env.reset(BOARD_AVAILABLE_2)
        self.assertEqual(set(self.env.available_moves()), {2})

        self.env.reset(BOARD_AVAILABLE_6)
        self.assertEqual(set(self.env.available_moves()), {6})

        self.env.reset(BOARD_AVAILABLE_NONE)
        self.assertEqual(set(self.env.available_moves()), set([]))

    def test_run_win_p1(self):
        env = self.env
        act_space = env.action_space.n
        moves1 = [i % act_space for i in range(100)]
        moves2 = [2 * i % act_space for i in range(1, 100)]
        p1 = DeterministicPlayer(env=env, moves=moves1, name="P1")
        p2 = DeterministicPlayer(env=env, moves=moves2, name="P2")
        res = env.run(p1, p2)
        self.assertEqual(ResultType.WIN1.value, res.value)
        self.assertEqual(moves1[:11], p1.action_log)
        self.assertEqual(moves2[:10], p2.action_log)
        self.assertEqual([ConnectFourEnv.DEF_REWARD] * 10 + [ConnectFourEnv.WIN_REWARD], p1.reward_log)
        self.assertEqual([ConnectFourEnv.DEF_REWARD] * 9 + [ConnectFourEnv.LOSS_REWARD], p2.reward_log)
        self.assertEqual([False] * 10 + [True], p1.done_log)
        self.assertEqual([False] * 9 + [True], p2.done_log)

        np.testing.assert_array_equal(p1.l_states[1], p1.states[1])
        np.testing.assert_array_equal(p1.l_new_states[0], p1.states[1])

        np.testing.assert_array_equal(p1.l_states[-1], p1.states[-1])
        np.testing.assert_array_equal(p1.l_new_states[-3], p1.states[-2])
        np.testing.assert_array_equal(p1.l_states[-2], p1.states[-2])

        np.testing.assert_array_equal(p2.l_new_states[0], p2.states[1])
        np.testing.assert_array_equal(p2.l_states[1], p2.states[1])

        np.testing.assert_array_equal(p2.l_states[-1], p2.states[-1])
        np.testing.assert_array_equal(p2.l_new_states[-3], p2.states[-2])
        np.testing.assert_array_equal(p2.l_states[-2], p2.states[-2])

    def test_run_win_p2(self):
        env = self.env
        act_space = env.action_space.n
        moves1 = [2 * i % act_space for i in range(100)]
        moves2 = [(2 * i + 1) % act_space for i in range(0, 100)]
        p1 = DeterministicPlayer(env=env, moves=moves1, name="P1")
        p2 = DeterministicPlayer(env=env, moves=moves2, name="P2")
        res = env.run(p1, p2)
        self.assertEqual(ResultType.WIN2.value, res.value)
        self.assertListEqual(moves1[:11], p1.action_log)
        self.assertListEqual(moves2[:11], p2.action_log)
        self.assertEqual([ConnectFourEnv.DEF_REWARD] * 10 + [ConnectFourEnv.LOSS_REWARD], p1.reward_log)
        self.assertEqual([ConnectFourEnv.DEF_REWARD] * 10 + [ConnectFourEnv.WIN_REWARD], p2.reward_log)
        self.assertEqual([False] * 10 + [True], p1.done_log)
        self.assertEqual([False] * 10 + [True], p2.done_log)

    def test_run_draw(self):
        random.seed(0)
        env = self.env
        act_space = env.action_space.n
        moves2 = [(2 * i + 1) % act_space for i in range(0, 100)]
        p1 = RandomPlayer(env=env, name="P1", seed=88)
        p2 = DeterministicPlayer(env=env, moves=moves2, name="P2")
        res = env.run(p1, p2)
        self.assertEqual(ResultType.DRAW.value, res.value)
        self.assertEqual([1, 3, 5, 0, 2, 4, 6, 1, 3, 5, 0, 2, 4, 6, 1, 3, 5, 0, 3, 4, 4], p2.action_log)
        self.assertEqual([ConnectFourEnv.DEF_REWARD] * 20 + [ConnectFourEnv.DRAW_REWARD], p2.reward_log)
        self.assertEqual([False] * 20 + [True], p2.done_log)

    def test_reset(self):
        env = self.env
        env.run(RandomPlayer(env=env, seed=0), RandomPlayer(env=env, seed=1))
        sum_steps = np.sum(np.sum(np.absolute(env.board)))
        self.assertEqual(17, sum_steps)
        env.reset()
        sum_steps = np.sum(np.sum(np.absolute(env.board)))
        self.assertEqual(0, sum_steps)
