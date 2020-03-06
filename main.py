from framework.data_manager import DataManager
from framework.frame import Frame
from framework.game import Game
from players.random import RandomPlayer
from players.policy_grad import PolicyGradPlayer
from players.static import PerfectPlayer

buffer_size = 300

data_manager = DataManager(max_size=buffer_size)
data_manager.get()

dense_player = PolicyGradPlayer('PolicyGrad', Frame.X)
player2 = RandomPlayer('Random', Frame.O)
player3 = PerfectPlayer('Static', Frame.O)

game = Game(dense_player, player2)

for i in range(5):
    for j in range(4):
        if j == 0:
            game.player_1 = dense_player
            game.player_2 = player2
        elif j == 1:
            game.player_1 = dense_player
            game.player_2 = player3
        elif j == 2:
            game.player_1 = player2
            game.player_2 = dense_player
        elif j == 3:
            game.player_1 = player3
            game.player_2 = dense_player
        game.start(10)
        data_manager.enqueue(game.matches)
        dense_player.train(50, data_manager)
        dense_player.save_params()
        game.matches.clear()
        # game.swap_players()

data_manager.write()
