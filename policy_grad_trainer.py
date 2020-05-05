from framework.data_manager import DataManager
from framework.frame import Frame
from framework.game import Game
from players.policy_grad import PolicyGradPlayer
from players.random import RandomPlayer


def keep_dense_learning():

    game = Game(
        PolicyGradPlayer('PolicyGrad', Frame.X),
        RandomPlayer('Random', Frame.O)
    )

    data_manager = DataManager(max_size=500)
    data_manager.get()

    dense_player = game.player_1

    for i in range(3):
        game.start(10)
        data_manager.enqueue(game.matches)
        dense_player.train(100, data_manager)
        dense_player.show_confusion_matrix(data_manager)
        game.matches.clear()
        game.swap_players()

        data_manager.write()


def main():
    keep_dense_learning()


if __name__ == '__main__':
    main()
