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
# player2 = PerfectPlayer('Static', Frame.O)

game = Game(dense_player, player2)

for i in range(10):
    game.start(10)
    data_manager.enqueue(game.matches)
    dense_player.train(50, data_manager)
    dense_player.save_params()
    game.matches.clear()
    game.swap_players()

data_manager.write()

