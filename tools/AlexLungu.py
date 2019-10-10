import numpy as np
from keras.engine.saving import load_model

from gym_connect_four import Player


class SavedPlayerCNN(object):
    pass


class AlexLunguPlayer(Player):
    def __init__(self, env, name='SavedPlayerCNN', model_prefix=None):
        super().__init__(env, name)

        if model_prefix is None:
            model_prefix = self.name

        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n

        print(f"Loading model from:{model_prefix}.h5")
        self.model = load_model(f"{model_prefix}.h5")

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space) + [1])
        for _ in range(100):
            q_values = self.model.predict(state)[0][0]
            q_values = np.array([[
                x if idx in self.env.available_moves() else -10
                for idx, x in enumerate(q_values[0])
            ]])
            action = np.argmax(q_values[0])
            if self.env.is_valid_action(action):
                return action

        raise Exception(
            'Unable to determine a valid move! Maybe invoke at the wrong time?'
        )
