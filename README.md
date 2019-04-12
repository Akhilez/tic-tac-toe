# Tic-Tac-Toe
TicTacToe game played by Neural Networks!

There are two parts to this.
1. Framework
2. Model

Framework contains the complete code to play the game. 

It has multiple types of players:
 - Human player: uses command line console for use inputs or some other UI
 - Random player: chooses a random position
 - Perfect player: a static algorithm that is designed to play as smart as a human
 - Dense network player: use deep neural network to pick a position
 - Cnn player: uses cnn to predict best position

Creating the framework is pretty straightforward and I will not discuss it here.

##### Inputs and outputs:
Every match between two players has varying number of inserts. The game board before inserting an x or o will be an input sample. And the position of insertion becomes the output.

##### Generalization:
I have designed the network so that it will ALWAYS try to predict a position for X and X only. But what if I have to predict a position for O? I will simply flip all Is to Os and Os to Xs and then predict for X. In this way, my X predictions will work for both Xs and Os.

##### Which insertions to train on?
In my initial design, I trained for all the insertions of the player who wins the game. This was a big mistake as my model barely won against the random player. The reason is that, if random won, then my model tries to train from the random data. For the next match, it again gets trained on random inserts.

There next design was to let perfect player and ann play and train for 1000 matches. Now the winner's insertions are perfect, so my model shall train on the best data. Then I would play against a random player and check if my model made progress.

##### Inputs Format
    x    = (1, 0, 0)
    o    = (0, 1, 0)
    None = (0, 0, 1)

##### Model design:
pending

TODO
  - Focus on parsing the weights from output to inputs.
  - Consider negative weights