
# gym-connect-four  
Open AI Gym for ConnectFour game  
  
### Setup  
  
Install the environment:  
  
``` bash  
git clone https://github.com/IASIAI/gym-connect-four.git  
cd gym-connect-four  
pip install -e .  
```  
  
Test two random players :  
  
``` bash  
python example/sample.opponent.py  
```  
  
### Usage  
  
Inside the environment there are a couple of sample players provided:  
* **RandomPlayer**: as name suggests it only does random moves, although valid ones  
* **SavedPlayer**: loads a saved model and uses it to play  

#### Sample code  
```python
import gym
from gym_connect_four import RandomPlayer, ConnectFourEnv
env: ConnectFourEnv = gym.make("ConnectFour-v0")

player1 = RandomPlayer(env, 'Dexter-Bot')
player2 = RandomPlayer(env, 'Deedee-Bot')
result = env.run(player1, player2, render=True)
reward = result.value
print(reward)
```
Inside the repo there are a couple of examples:  
* **sample_nn**: Neural Network implementation identical to the one from CartPole playing against a random opponent  
  
Considerations for the environment:  
  
* the environment will throw an exception on invalid move, please use either ```env
.is_valid_action(action)``` or pick the move from ```env.available_moves()```
* Player gets the board through `get_next_action()` and `learn()` publicly exposed methods. **Every**
 player sees its discs **encoded with 1**, **regardless** his turn order in the game. In this way
 , the same NN model can be used as Player1 or as Player2 without needing to adjust board encoding.

### Assignment  
  
For the competition the following set of deliverables must be provided:  
* python file containing a Player class used for training the model (NNPlayer + DQNSolver if relative to sample_nn.py)  
* python class containing a SavedPlayer implementation (only if any customization was needed)  
* a .h5 (HDF5) file containing the model structure, weights and optimizer (will be the provided or default SavedPlayer class) (https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)  
  
### Evaluation of the assignment  
  
All supplied models will pitted against each other in a round robbin tournament. Every player will play N (a big number like 1000) rounds with every other player. The formula of the score:

Score = sum( (**#Wins** against *Player<sub>k</sub>*) - (**#Losses** against *Player<sub>k</sub>*) for *k* in *{all_opponents}*)

The draws are not taken into account for score computation.
The winner is the one having highest score.
  
### Have fun and happy training!
