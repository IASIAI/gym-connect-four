import gym
from gym_connect_four import RandomPlayer

env = gym.make('ConnectFour-v0')

player1 = RandomPlayer(env, 'RandomPlayer1')
player2 = RandomPlayer(env, 'RandomPlayer2')

total_reward = 0
done = False
state = env.reset()
MAX_INVALID_MOVES = 50
while not done:
    for player in [player1, player2]:

        # Loop is here in case Agent generates invalid moves itself.
        for _ in range(MAX_INVALID_MOVES):
            action = player.get_next_action(state)
            if env.is_valid_action(action):
                break
        else:
            raise Exception('Unable to determine a valid move! Maybe invoke at the wrong time?')

        new_state, reward, done, _ = env.step(action)

        player.learn(state, action, reward, new_state, done)

        total_reward += reward
        if done:
            if reward == 1:
                print(f"winner: {player.name}")
                print("board state:\n", new_state)
                print(f"total reward={total_reward}")
            else:
                print(f"draw after {player.name} move")
                print("board state:\n", new_state)
                print(f"total reward={total_reward}")
            break

        state = new_state

env.close()
