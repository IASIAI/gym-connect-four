import os

import keras
import numpy as np

from gym_connect_four import Player


class ValiRoscaPlayer(Player):
    def __init__(self, env, name='RandomPlayer'):
        super(ValiRoscaPlayer, self).__init__(env, name)
        self.observation_space = env.observation_space.shape
        self.action_space = env.action_space.n
        self.first_player_inference_model = None
        self.second_player_inference_model = None
        self.load_model(".")

    def restricted_argmax(self, array, limited_indexes):
        limited_indexes = list(limited_indexes)
        best_idx = limited_indexes[0]
        if len(limited_indexes) > 1:
            for idx in limited_indexes[1:]:
                if array[0][idx] > array[0][best_idx]:
                    best_idx = idx

        return best_idx

    def get_next_action(self, state: np.ndarray) -> int:
        state = np.reshape(state, [1] + list(self.observation_space))
        for _ in range(100):
            if np.sum(state != 0) % 2 == 0:
                prediction = self.first_player_inference_model.predict(state)
            else:
                prediction = self.second_player_inference_model.predict(state)

            action = self.restricted_argmax(prediction, self.env.available_moves())
            if self.env.is_valid_action(action):
                return action

        raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')

    def learn(self, state, action, state_next, reward, done):
        pass

    def episode_end(self, *args, **kwargs):
        pass

    def load_model(self, model_prefix: str = None):
        self.first_player_inference_model = keras.models.load_model(
            os.path.join(model_prefix, 'Valentin_Rosca_first_wizardly_dubinsky_inference.h5'))
        self.second_player_inference_model = keras.models.load_model(
            os.path.join(model_prefix, 'Valentin_Rosca_second_fervent_volhard_inference.h5'))
