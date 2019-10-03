
  
# gym-connect-four Open AI Gym for ConnectFour game    
    
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
    
Inside the repo there are a couple of examples:    
* **sample_nn**: Neural Network implementation identical to the one from CartPole playing against a random opponent    
* **sample_opponent**: Simple random vs random    
* **sample_two_players**: Using two defined players, without relying on the opponent feature in the environment    
    
Considerations for the environment:    
* the ```reset``` function is the one that assigns the opponent and must be specified each time, unless both players are handled by user (as in sample_two_players)    
* the environment will throw an exception on invalid move, please use either ```env.is_valid_action(action)``` or pick the move from ```env.available_moves```  

### Runner usage
In the ```tools``` directory there is a script called runner.py which will be the tool used to execute the final competition.
``` bash
cd tools
python runner.py random Model playerclass random
```
It accepts a list of arguments, from which only the ```random``` can repeat itself.
Each imput parameter will be validated in order against:
* "random": It will instantiate a player based on the RandomPlayer class, but with a randomized name
* "parameter.py": If a python file with the same name is present, it will import that file and instantiate a class with Parameter name (capitalized). In that class you will be responsable to load your model in the init section and then to supply the requested move (maybe to reformat the board when requested for your specific model)
* "Parameter.h5": If your model is saved from a simple Neural Network and you haven't altered the input board on processing, then the model will be loaded via the SavedPlayer class

The runner will pit each player against all the others and will also allow learning for all of them.

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