import argparse
import sys
import os.path
import importlib
import random

import gym
from gym_connect_four import RandomPlayer, SavedPlayer

env = gym.make("ConnectFour-v0")
ROUNDS = 50
LEARNING = True
DISPLAY_LEADERBOARD_EACH_ROUND = False


def tournament_player_loader(model: str):
    if model == "random":
        return RandomPlayer(env, name=f"RandomPlayer{random.randint(1,999)}")

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
        player1.reset(episode, 1)
        player2.reset(episode, -1)

        done = False
        player2_learn = False
        while not done:
            action1 = player1.get_next_action(state)
            state1, reward1, done1, _ = env.step(action1)

            if player2_learn:
                if LEARNING:
                    player2.learn(state1, action2, env._reverse_reward(reward1), state, done1)
            else:
                player2_learn = True

            if not done1:
                action2 = player2.get_next_action(state1)
                state2, reward2, done2, _ = env.step(action2)

                if LEARNING:
                    player1.learn(state, action1, env._reverse_reward(reward2), state2, done2)

                if done2:
                    done = True
                    if LEARNING:
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
                if LEARNING:
                    player1.learn(state, action1, reward1, state1, done1)

                if reward1 != env.DRAW_REWARD:
                    # player1 Won
                    match_result = 1
                else:
                    # player1 Draw
                    match_result = 0

        # idx0 = Win, idx1 = Draw, idx2 = Loss
        results[1 - match_result] += 1
        print(f"{player1.name}:{player2.name}={results}")

    return results


def tournament_print(leaderboard):
    for item in leaderboard:
        leaderboard[item][4] = tournament_score(leaderboard[item])

    leaderboard = sorted(leaderboard.items(), reverse=True, key=lambda itm: itm[1][4])

    print("{:6s} {:25s} {:7s} | {:23s} | {:23s} | {:23s} | {:23s}".format("", "", "", "Starter Rounds", "Second Rounds", "Starter matches", "Second matches"))

    print("{:6s} {:25s} {:7s} | {:7s} {:7s} {:7s} | {:7s} {:7s} {:7s} | {:7s} {:7s} {:7s} | {:7s} {:7s} {:7s}".format("Place", "Name", "Score", "Wins", "Draws", "Losses", "Wins", "Draws", "Losses", "Wins", "Draws", "Losses", "Wins", "Draws", "Losses"))
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


def tournament(models):
    players = []
    for model in models:
        players.append(tournament_player_loader(model))

    print("Loaded", len(players), "players :", ", ".join([player.name for player in players]))

    game_list = []
    leaderboard = {}

    for player1 in players:
        leaderboard[player1.name] = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], 0]
        for player2 in players:
            if player1 == player2:
                continue
            game_list.append((player1, player2))

    random.shuffle(game_list)

    for game in game_list:
        score = play_game(game[0], game[1], ROUNDS)
        print(score)

        p1 = game[0].name
        p2 = game[1].name
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

    for player in players:
        player.save_model()


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
