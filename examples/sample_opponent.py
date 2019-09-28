import gym
from gym_connect_four import RandomPlayer

EPISODES = 10
MAX_INVALID_MOVES = 50

env = gym.make('ConnectFour-v0')

player = RandomPlayer(env, 'RandomPlayer1')
opponent = RandomPlayer(env, 'OpponentRandomPlayer')

for episode in range(EPISODES):
    # Player color is 1 if starting, or -1 if you are second player. It's not the opponent's color
    state = env.reset(opponent=opponent, player_color=1)
    done = False
    total_reward = 0
    while not done:
        # Loop is here in case Agent generates invalid moves itself.
        for _ in range(MAX_INVALID_MOVES):
            action = player.get_next_action(state)
            if env.is_valid_action(action):
                break
        else:
            raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')

        new_state, reward, done, _ = env.step(action)

        player.learn(new_state, action, reward, done)

        if done:
            print(f"\nepisode: {episode}")
            if reward == 1:
                print(f"winner: {player.name}")
                print("board state:\n", new_state)
                print(f"reward={reward}")
            elif reward == env.LOSS_REWARD:
                print(f"lost to: {opponent.name}")
                print("board state:\n", new_state)
                print(f"reward={reward}")
            elif reward == env.DRAW_REWARD:
                print(f"draw after {player.name} move")
                print("board state:\n", new_state)
                print(f"reward={reward}")
            break

        state = new_state

env.close()
