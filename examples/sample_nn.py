import random
import warnings
from collections import deque

import gym
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from tqdm import tqdm

from gym_connect_four import RandomPlayer, ConnectFourEnv, Player, ResultType

ENV_NAME = "ConnectFour-v0"
TRAIN_EPISODES = 100

class DQNSolver:
    """
    Vanilla Multi Layer Perceptron version
    """

    def __init__(self, observation_space, action_space):
        self.GAMMA = 0.95
        self.LEARNING_RATE = 0.001

        self.MEMORY_SIZE = 1000000
        self.BATCH_SIZE = 20

        self.EXPLORATION_MAX = 1.0
        self.EXPLORATION_MIN = 0.01
        self.EXPLORATION_DECAY = 0.995

        self.exploration_rate = self.EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=self.MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Flatten(input_shape=observation_space))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_moves=[]):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        q_values = np.array([[x if idx in available_moves else -100 for idx, x in enumerate(q_values[0])]])
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        batch = random.sample(self.memory, self.BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= self.EXPLORATION_DECAY
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)

    def save_model(self, file_prefix: str):
        self.model.save(f"{file_prefix}.h5")


class NNPlayer(Player):
    def __init__(self, env, name='RandomPlayer'):
        super(NNPlayer, self).__init__(env, name)

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        self.dqn_solver = DQNSolver(self.observation_space, self.action_space)

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        for _ in range(100):
            action = self.dqn_solver.act(state, self.env.available_moves())
            if self.env.is_valid_action(action):
                return action
        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')

    def learn(self, state, action, state_next, reward, done) -> None:
        state = np.reshape(state, [1] + list(self.observation_space))
        state_next = np.reshape(state_next, [1] + list(self.observation_space))

        # reward = reward if not done else -reward
        self.dqn_solver.remember(state, action, reward, state_next, done)

        if not done:
            self.dqn_solver.experience_replay()

    def save_model(self):
        self.dqn_solver.save_model(self.name)


def game(show_boards=False):
    env: ConnectFourEnv = gym.make(ENV_NAME)

    player = NNPlayer(env, 'NNPlayer')
    opponent = RandomPlayer(env, 'OpponentRandomPlayer')

    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    for run in tqdm(range(1, TRAIN_EPISODES + 1)):
        env.reset()
        result = env.run(player, opponent, render=False)
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
    print(f"Wins [{wins}], Draws [{draws}], Losses [{losses}] - Total reward {total_reward}, average reward {total_reward / TRAIN_EPISODES}")

    player.save_model()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        game(False)
