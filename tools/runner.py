import argparse
import sys
import os.path
import importlib

import gym
from gym_connect_four import RandomPlayer, SavedPlayer

env = gym.make("ConnectFour-v0")
ROUNDS = 200


def tournament_player_loader(model: str):
    if model == "random":
        return RandomPlayer(env, name="RandomPlayer")

    if os.path.exists(f"{model}.py"):
        module = importlib.import_module(model)
        class_ = getattr(module, model.capitalize())
        player = class_(env, name=model.capitalize())
        return player

    if os.path.exists(f"{model}.h5"):
        return SavedPlayer(env, name=model, model_prefix=model)

    raise Exception(f"Unable to handle module {model}")


def play_game(player1, player2, rounds=ROUNDS):
    print(f"{player1.name} vs {player2.name}")
    results = [0] * 3
    for episode in range(rounds):
        match_result = None  # 1 = win, 0 = draw, -1 = loss

        state = env.reset()
        done = False
        player2_learn = False
        while not done:
            action1 = player1.get_next_action(state)
            state1, reward1, done1, _ = env.step(action1)

            if player2_learn:
                player2.learn(state1, action2, env._reverse_reward(reward1), state, done1)
            else:
                player2_learn = True

            if not done1:
                action2 = player2.get_next_action(state1)
                state2, reward2, done2, _ = env.step(action2)

                player1.learn(state, action1, env._reverse_reward(reward2), state2, done2)

                if done2:
                    done = True
                    player2.learn(state1, action2, reward2, state2, done2)
                    if reward2 != env.DRAW_REWARD:
                        # player2 Won
                        match_result = -1
                    else:
                        # player2 Draw
                        match_result = 0

                state = state2
            else:
                done = True
                player1.learn(state, action1, reward1, state1, done1)
                if reward1 != env.DRAW_REWARD:
                    # player1 Won
                    match_result = 1
                else:
                    # player1 Draw
                    match_result = 0

        # idx0 = Win, idx1 = Draw, idx2 = Loss
        results[1-match_result] += 1
        #print(f"{player1.name}:{player2.name}={match_result}")

    return results


def tournament(models):
    players = []
    for model in models:
        players.append(tournament_player_loader(model))

    print("Loaded", len(players),  "players :", ", ".join([player.name for player in players]))

    game_list = []
    for player1 in players:
        for player2 in players:
            if player1 == player2:
                continue
            game_list.append((player1, player2))

    for game in game_list:
        score = play_game(game[0], game[1], ROUNDS)
        print(score)


def main():
    parser = argparse.ArgumentParser(description="Tournament runner for connect four")
    parser.add_argument("models", metavar="model", type=str, nargs="+", help="List of models")
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    tournament(args.models)


if __name__ == "__main__":
    main()
