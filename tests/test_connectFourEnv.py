from unittest import TestCase
from gym_connect_four import RandomPlayer

import gym

BOARD_VALIDATION = [[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0],
                    [0, 0, -1, 1, 0, -1, 0],
                    [0, 0, 1, 1, 0, 1, 1],
                    [-1, -1, -1, -1, 0, -1, -1],
                    [1, 1, -1, 1, -1, 1, 1]]

BOARD_WIN_ROW = [[0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 0, -1, 0, 0, 0],
                 [0, 0, -1, 1, 0, -1, 0],
                 [0, 0, 1, 1, 0, 1, 1],
                 [-1, -1, -1, -1, 0, -1, -1],
                 [1, 1, -1, 1, -1, 1, 1]]

BOARD_WIN_COLUMN = [[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, -1, 1, 0, -1, 0],
                    [0, 0, 1, 1, 0, 1, 1],
                    [-1, 0, -1, -1, 0, -1, -1],
                    [1, 1, -1, 1, -1, 1, 1]]

BOARD_WIN_DIAGONAL = [[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0],
                      [0, 0, -1, 1, 0, -1, 0],
                      [0, 0, 1, 1, 1, 1, 1],
                      [-1, 1, -1, -1, 0, 1, -1],
                      [1, 1, -1, 1, -1, 1, 1]]

BOARD_WIN_BDIAGONAL = [[0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 1, 0, 0, 0],
                       [0, 0, -1, 1, 0, -1, 0],
                       [0, 0, 1, 1, 0, 1, 1],
                       [-1, 1, -1, -1, 0, -1, -1],
                       [1, 1, -1, 1, -1, 1, 1]]

BOARD_AVAILABLE_0123 = [[0, 0, 0, 0, -1, 1, -1],
                        [0, 0, 0, 1, 1, -1, 1],
                        [0, 0, -1, 1, 1, -1, -1],
                        [0, 0, 1, 1, 1, 1, 1],
                        [-1, 1, -1, -1, -1, -1, -1],
                        [1, 1, -1, 1, -1, 1, 1]]

BOARD_AVAILABLE_2 = [[1, 1, 0, -1, -1, 1, -1],
                     [1, 1, -1, 1, 1, -1, 1],
                     [1, 1, -1, 1, 1, -1, -1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [-1, 1, -1, -1, -1, -1, -1],
                     [1, 1, -1, 1, -1, 1, 1]]

BOARD_AVAILABLE_6 = [[1, 1, 1, 1, -1, 1, 0],
                     [1, 1, -1, 1, 1, -1, 1],
                     [1, 1, -1, 1, 1, -1, -1],
                     [1, 1, 1, 1, 1, 1, 1],
                     [-1, 1, -1, -1, -1, -1, -1],
                     [1, 1, -1, 1, -1, 1, 1]]

BOARD_AVAILABLE_NONE = [[1, 1, 1, 1, -1, 1, 1],
                        [1, 1, -1, 1, 1, -1, 1],
                        [1, 1, -1, 1, 1, -1, -1],
                        [1, 1, 1, 1, 1, 1, 1],
                        [-1, 1, -1, -1, -1, -1, -1],
                        [1, 1, -1, 1, -1, 1, 1]]


class TestConnectFourEnv(TestCase):

    def setUp(self) -> None:
        self.env = gym.make('ConnectFour-v0')

    def test_step(self):
        pass

    def test_reset(self):
        pass

    def test_is_valid_action(self):
        self.env = gym.make('ConnectFour-v0')
        self.env.reset()
        self.env.board = BOARD_VALIDATION
        self.assertTrue(self.env.is_valid_action(0))
        self.assertFalse(self.env.is_valid_action(3))

    def test_is_win_state(self):
        self.env = gym.make('ConnectFour-v0')
        self.env.board = BOARD_WIN_ROW
        self.assertTrue(self.env.is_win_state())

        self.env.board = BOARD_WIN_COLUMN
        self.assertTrue(self.env.is_win_state())

        self.env.board = BOARD_WIN_DIAGONAL
        self.assertTrue(self.env.is_win_state())

        self.env.board = BOARD_WIN_BDIAGONAL
        self.assertTrue(self.env.is_win_state())

    def test_available_moves(self):
        self.env = gym.make('ConnectFour-v0')
        self.env.board = BOARD_AVAILABLE_0123
        self.assertEqual(set(self.env.available_moves()), set([0, 1, 2, 3]))

        self.env.board = BOARD_AVAILABLE_2
        self.assertEqual(set(self.env.available_moves()), set([2]))

        self.env.board = BOARD_AVAILABLE_6
        self.assertEqual(set(self.env.available_moves()), set([6]))

        self.env.board = BOARD_AVAILABLE_NONE
        self.assertEqual(set(self.env.available_moves()), set([]))
