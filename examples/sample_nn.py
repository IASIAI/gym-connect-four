import os
from operator import itemgetter

# Comment next line to run on GPU. With this configuration, it looks to run faster on CPU i7-8650U
os.environ['CUDA_VISIBLE_D5EVICES'] = '-1'

import random
import warnings

import gym
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from gym_connect_four import RandomPlayer, ConnectFourEnv, Player, ResultType

ENV_NAME = "ConnectFour-v0"
TRAIN_EPISODES = 100000

import threading
import time
from statistics import mean

import matplotlib
from plotly.subplots import make_subplots

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np
import pandas as pd
from plotly import graph_objects as go

SCORES_CSV_PATH = "./scores/scores.csv"
SCORES_PNG_PATH = "./scores/scores.png"
SOLVED_CSV_PATH = "./scores/solved.csv"
SOLVED_PNG_PATH = "./scores/solved.png"
AVERAGE_SCORE_TO_SOLVE = 195
CONSECUTIVE_RUNS_TO_SOLVE = 200
PLOT_REFRESH = 50


class ScoreLogger:

    def __init__(self, env_name):
        self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.averages = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.last_20_avg = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.last20_scores = deque(maxlen=20)
        self.exp_rates = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.time_hist = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
        self.t1 = time.time()

        self.env_name = env_name

        if os.path.exists(SCORES_PNG_PATH):
            os.remove(SCORES_PNG_PATH)
        if os.path.exists(SCORES_CSV_PATH):
            os.remove(SCORES_CSV_PATH)

    def show_graph(self, y: pd.DataFrame):
        self.fig = make_subplots(specs=[[{"secondary_y": True}]])
        self.fig.add_trace(go.Scatter(x=y.index, y=y.score, name="score"))
        self.fig.add_trace(go.Scatter(x=y.index, y=y.m, name="mean"))
        self.fig.add_trace(go.Scatter(x=y.index, y=y.m20, name="mean_last20"))
        self.fig.add_trace(go.Scatter(x=y.index, y=y.expl, name="expl"))
        self.fig.add_trace(go.Scatter(x=y.index, y=y.time, name="time"), secondary_y=True)
        self.fig.show()

    def add_score(self, score: int, run: int, exploration_rate: float, memory_size: int, refresh=False):

        self._save_csv(SCORES_CSV_PATH, score)
        self._save_png(input_path=SCORES_CSV_PATH,
                       output_path=SCORES_PNG_PATH,
                       x_label="runs",
                       y_label="scores",
                       average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                       show_goal=True,
                       show_trend=True,
                       show_legend=True)
        self.scores.append(score)
        self.last20_scores.append(score)
        last_20mean = mean(self.last20_scores)
        self.last_20_avg.append(last_20mean)
        mean_score = mean(self.scores)
        self.averages.append(mean_score)
        self.exp_rates.append(exploration_rate)
        td = time.time() - self.t1
        self.time_hist.append(td)

        if refresh:
            # Here we start a new thread as because of a bug in Plotly, sometimes the fig.show() doesn't return at all and process freezes
            y = pd.DataFrame(zip(self.scores, self.averages, self.last_20_avg, self.exp_rates, self.time_hist),
                             columns=['score', 'm', 'm20', 'expl', 'time'])

            threading.Thread(target=self.show_graph, args=(y,)).start()
        print(f"Run {run:3}: (avg: {mean_score:2.3f}, last20_avg: {last_20mean:2.3f}, expl: {exploration_rate:1.3}, "
              f"mem_sz: {memory_size!s}, time: {td:3.1})\n")
        if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
            solve_score = run - CONSECUTIVE_RUNS_TO_SOLVE
            print("Solved in " + str(solve_score) + " runs, " + str(run) + " total runs.")
            self._save_csv(SOLVED_CSV_PATH, solve_score)
            self._save_png(input_path=SOLVED_CSV_PATH,
                           output_path=SOLVED_PNG_PATH,
                           x_label="trials",
                           y_label="steps before solve",
                           average_of_n_last=None,
                           show_goal=False,
                           show_trend=False,
                           show_legend=False)
            exit()

    def _save_png(self, input_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            j = 0
            for i in range(0, len(data)):
                if len(data[i]) == 0:
                    continue
                x.append(int(j))
                y.append(int(data[i][0]))
                j += 1

        plt.subplots()
        plt.plot(x, y, label="score per run")

        average_range = average_of_n_last if average_of_n_last is not None else len(x)
        plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle="--",
                 label="last " + str(average_range) + " runs average")

        if show_goal:
            plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=":", label=str(AVERAGE_SCORE_TO_SOLVE) + " score average goal")

        if show_trend and len(x) > 1:
            trend_x = x[1:]
            z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
            p = np.poly1d(z)
            plt.plot(trend_x, p(trend_x), linestyle="-.", label="trend")

        plt.title(self.env_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w"):
                pass
        scores_file = open(path, "a")
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])


class DQNSolver:
    """
    Vanilla Multi Layer Perceptron version
    """

    def __init__(self, observation_space, action_space):
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.001

        self.MEMORY_SIZE = 512
        self.BATCH_SIZE = 32

        self.EXPLORATION_MAX = 1.0
        self.EXPLORATION_MIN = 0.0
        self.EXPLORATION_DECAY = 0.995

        self.exploration_rate = self.EXPLORATION_MAX
        self.isFit = False

        self.action_space = action_space
        self.memory = deque(maxlen=self.MEMORY_SIZE)
        obs_space_card = observation_space[0] * observation_space[1]
        self.model = Sequential()
        self.model.add(Flatten(input_shape=observation_space))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(obs_space_card * 2, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_moves):
        if np.random.rand() < self.exploration_rate:
            return random.choice(list(available_moves))
        q_values = self.model.predict(state)[0]
        vs = [(i, q_values[i]) for i in available_moves]
        act = max(vs, key=itemgetter(1))
        return act[0]

    def experience_replay(self):
        if self.isFit:
            self.exploration_rate *= self.EXPLORATION_DECAY
            self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)

        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        if not self.isFit:
            states = list(map(lambda _: _[0][0], batch))
            states = np.array(states)
            self.model.fit(states, np.zeros((len(batch), self.action_space)), verbose=0)

        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.isFit = True

    def save_model(self, file_prefix: str):
        self.model.save(f"{file_prefix}.h5")


class NNPlayer(Player):
    def __init__(self, env, name='RandomPlayer'):
        super(NNPlayer, self).__init__(env, name)

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.dqn_solver = DQNSolver(self.observation_space, self.action_space)
        self.sl = ScoreLogger(str(self.__class__))
        self._N = 30
        self._STOP_THRESHOLD = 0.9
        self._last_N_rounds = deque(maxlen=self._N)
        self._round = 0
        self._score = 0
        self._total_score = 0

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        action = self.dqn_solver.act(state, self.env.available_moves())
        if self.env.is_valid_action(action):
            return action

    def _stop_learn_condition(self):
        return len(self._last_N_rounds) == self._N and mean(self._last_N_rounds) >= self._STOP_THRESHOLD

    def learn(self, state, action, state_next, reward, done) -> None:
        if self._stop_learn_condition():
            print(f"Stopping learning as got {mean(self._last_N_rounds)} avg on last{self._N}. Saving model & exiting")
            self.save_model()
            exit()
        state = np.reshape(state, [1] + list(self.observation_space))
        state_next = np.reshape(state_next, [1] + list(self.observation_space))

        # reward = reward if not done else -reward
        self.dqn_solver.remember(state, action, reward, state_next, done)

        self.dqn_solver.experience_replay()
        if done:
            self._last_N_rounds.append(int(reward))
            self._round += 1
            self._total_score += int(reward)
            self.sl.add_score(int(reward), self._round, self.dqn_solver.exploration_rate, len(self.dqn_solver.memory),
                              refresh=self._round % PLOT_REFRESH == 0)

    def save_model(self):
        self.dqn_solver.save_model(self.name)


def game(show_boards=False):
    env: ConnectFourEnv = gym.make(ENV_NAME)

    player = NNPlayer(env, 'NNPlayer')
    opponent = RandomPlayer(env, 'OpponentRandomPlayer')
    players = [player, opponent]

    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    for run in range(1, TRAIN_EPISODES + 1):
        random.shuffle(players)
        result = env.run(*players, None, render=False)
        reward = result.value
        total_reward += reward

        wins += max(0, result.value)
        losses += max(0, -result.value)
        draws += (abs(result.value) + 1) % 2

        if show_boards:
            print("Run: " + str(run) + ", score: " + str(reward))
            if hasattr(player, 'dqn_solver'):
                print("exploration: " + str(player.dqn_solver.exploration_rate))
            if result == ResultType.WIN1:
                print(f"winner: {player.name}")
                print("board state:\n", env.board)
                print(f"reward={reward}")
            elif result == ResultType.WIN2:
                print(f"lost to: {opponent.name}")
                print("board state:\n", env.board)
                print(f"reward={reward}")
            elif result == ResultType.DRAW:
                print(f"draw after {player.name} move")
                print("board state:\n", env.board)
                print(f"reward={reward}")
            else:
                raise ValueError("Unknown result type")
    print(
        f"Wins [{wins}], Draws [{draws}], Losses [{losses}] - Total reward {total_reward}, average reward {total_reward / TRAIN_EPISODES}")

    player.save_model()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        game(False)
