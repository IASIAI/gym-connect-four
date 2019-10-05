import argparse
import importlib
import os.path
import random
import sys
from typing import List

import gym

from gym_connect_four import RandomPlayer, SavedPlayer, ConnectFourEnv, Player

env: ConnectFourEnv = gym.make("ConnectFour-v0")
ROUNDS = 50
LEARNING = True
DISPLAY_LEADERBOARD_EACH_ROUND = True


def tournament_player_loader(model: str):
    if model == "random":
        return RandomPlayer(env, name=f"RandomPlayer{random.randint(1, 999)}")

    if os.path.exists(f"{model}.py"):
        module = importlib.import_module(model)
        class_ = getattr(module, model.capitalize())
        player = class_(env, name=model.capitalize())
        return player

    if os.path.exists(f"{model}.h5"):
        return SavedPlayer(env, name=model, model_prefix=model)

    raise Exception(f"Unable to handle module {model}")


def play_game(player1, player2, rounds=ROUNDS) -> List[int]:
    print(f"{player1.name} vs {player2.name}")

    results = [0] * 3
    for episode in range(rounds):
        result = env.run(player1, player2)
        match_result: int = result.value  # 1 = win, 0 = draw, -1 = loss
        results[1 - match_result] += 1
        print(f"{player1.name}:{player2.name}={results}")

    return results


def tournament_print(leaderboard):
    for item in leaderboard:
        leaderboard[item][4] = tournament_score(leaderboard[item])

    leaderboard = sorted(leaderboard.items(), reverse=True, key=lambda itm: itm[1][4])

    print("{:6s} {:25s} {:7s} | {:23s} | {:23s} | {:23s} | {:23s}".format("", "", "", "Starter Rounds", "Second Rounds", "Starter matches",
                                                                          "Second matches"))

    print("{:6s} {:25s} {:7s} | {:7s} {:7s} {:7s} | {:7s} {:7s} {:7s} | {:7s} {:7s} {:7s} | {:7s} {:7s} {:7s}".format("Place", "Name",
                                                                                                                      "Score", "Wins",
                                                                                                                      "Draws", "Losses",
                                                                                                                      "Wins", "Draws",
                                                                                                                      "Losses", "Wins",
                                                                                                                      "Draws", "Losses",
                                                                                                                      "Wins", "Draws",
                                                                                                                      "Losses"))
    idx = 0
    for item in leaderboard:
        idx += 1
        print(
            "{:6d} {:25s} {:7d} | {:7d} {:7d} {:7d} | {:7d} {:7d} {:7d} | {:7d} {:7d} {:7d} | {:7d} {:7d} {:7d}".
                format(idx, item[0], item[1][4],
                       item[1][0][0], item[1][0][1], item[1][0][2],
                       item[1][1][0], item[1][1][1], item[1][1][2],
                       item[1][2][0], item[1][2][1], item[1][2][2],
                       item[1][3][0], item[1][3][1], item[1][3][2]
                       ))
    pass


def tournament_score(score):
    # Score = Number of wins -  Number of losses
    return (score[0][0] - score[0][2]) + (score[1][0] - score[1][2])


def tournament(players: List[Player], save_models: bool = False):
    game_list = []
    leaderboard = {}

    for player1 in players:
        leaderboard[player1.name] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0]
        for player2 in players:
            if player1 == player2:
                continue
            game_list.append((player1, player2))

    random.shuffle(game_list)

    for player1, player2 in game_list:
        score = play_game(player1, player2, ROUNDS)
        print(score)

        p1 = player1.name
        p2 = player2.name
        leaderboard[p1][0] = [x + y for x, y in zip(leaderboard[p1][0], score)]
        leaderboard[p2][1] = [x + y for x, y in zip(leaderboard[p2][1], score[::-1])]
        if score[0] > score[2]:
            leaderboard[p1][2][0] += 1
            leaderboard[p2][3][2] += 1
        elif score[0] == score[2]:
            leaderboard[p1][2][1] += 1
            leaderboard[p2][3][1] += 1
        else:
            leaderboard[p1][2][2] += 1
            leaderboard[p2][3][0] += 1

        if DISPLAY_LEADERBOARD_EACH_ROUND:
            tournament_print(leaderboard)

    if not DISPLAY_LEADERBOARD_EACH_ROUND:
        tournament_print(leaderboard)

    if save_models:
        for player in players:
            player.save_model()


def main():
    description = ("Tournament runner for connect four\n"
                   "Requires at least 2 models. Example:"
                   "runner.py random vlada NNPlayer")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("models", metavar="model", type=str, nargs="+",
                        help="List of models. Accepts [<*.py filename>|<H5 filename>|random]")
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    players = [tournament_player_loader(model) for model in args.models]

    print("Loaded", len(players), "players :", ", ".join([player.name for player in players]))

    tournament(players)


if __name__ == "__main__":
    main()
    # tournament([RandomPlayer(env, name="R1", seed=0), RandomPlayer(env, name="R2", seed=0), SavedPlayer(env, "NNPlayer")])
