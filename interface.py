from NeuralNetworks.TicTacToe.framework.data_manager import DataManager
from NeuralNetworks.TicTacToe.framework.frame import Frame
from NeuralNetworks.TicTacToe.framework.game import Game
from NeuralNetworks.TicTacToe.players.convolutional import ConvolutionalPlayer
from NeuralNetworks.TicTacToe.players.dense import DenseNetworkPlayer
from NeuralNetworks.TicTacToe.players.human import HumanPlayer
from NeuralNetworks.TicTacToe.players.random import RandomPlayer
from NeuralNetworks.TicTacToe.players.static import StaticPlayer


class TicTacToe:

    @staticmethod
    def create_console_game():
        """
        Read player 1 details
        Read player 2 details
        :return:
        """
        print("\nPlayer 1:")
        player_1_type = TicTacToe.read_player_type()
        player_1_name = input('Enter player 1 name: ')
        player_1_character = TicTacToe.read_character()

        print("\nPlayer 2:")
        player_2_type = TicTacToe.read_player_type()
        player_2_name = input('Enter player 2 name: ')

        TicTacToe.create_game(player_1_name, player_1_type, player_2_name, player_2_type, player_1_character).start()

    @staticmethod
    def create_automated_game(type_1, type_2, num_matches=1):
        game = TicTacToe.create_game(type_1, type_1, type_2, type_2, Frame.O)
        game.start(num_matches)
        with DataManager() as data_manager:
            data_manager.data = game.matches

    @staticmethod
    def read_character():
        while True:
            character = input('Enter the player\'s character (X or O): ').upper()
            if character == Frame.X or character == Frame.O:
                return character
            print(f'Please enter either {Frame.X} or {Frame.O}')

    @staticmethod
    def create_game(player1_name=None, player1_type=HumanPlayer.TYPE, player2_name=None,
                    player2_type=HumanPlayer.TYPE, player1_character=Frame.X):

        player2_character = Frame.X if player1_character == Frame.O else Frame.O

        player1 = TicTacToe.get_player(player1_type, player1_name, player1_character)
        player2 = TicTacToe.get_player(player2_type, player2_name, player2_character)

        return Game(player1, player2)

    @staticmethod
    def get_player(player_type, player_name, player_character):
        if player_type == HumanPlayer.TYPE:
            return HumanPlayer(player_name, player_character)
        if player_type == RandomPlayer.TYPE:
            return RandomPlayer(player_name, player_character)
        if player_type == DenseNetworkPlayer.TYPE:
            return DenseNetworkPlayer(player_name, player_character)
        if player_type == StaticPlayer.TYPE:
            return StaticPlayer(player_name, player_character)
        if player_type == ConvolutionalPlayer.TYPE:
            return ConvolutionalPlayer(player_name, player_character)
        raise Exception(f"Player type {player_type} not found!")

    @staticmethod
    def read_player_type():
        while True:
            character = input('\n1. Human\n2. Randon\n3. Dense\nEnter the player type: ')
            if character not in '123':
                print("Wrong input")
            else:
                return {'1': HumanPlayer.TYPE, '2': RandomPlayer.TYPE, '3': DenseNetworkPlayer.TYPE}[character]

    @staticmethod
    def keep_dense_learning():

        buffer_size = 1000

        game = Game(
            ConvolutionalPlayer('Conv_1', Frame.X),
            # DenseNetworkPlayer('Dense_1', Frame.X),
            # DenseNetworkPlayer('Dense_1', Frame.O)
            RandomPlayer('Random', Frame.O)
            # StaticPlayer('Static', Frame.O)
            # HumanPlayer('Human', Frame.O)
        )

        data_manager = DataManager(max_size=buffer_size)

        dense_player = game.player_1

        for i in range(1000):
            game.start(1)
            # if game.current_match.winner is not None:
            data_manager.enqueue(game.matches)
            dense_player.model.train(15, data_manager)
            game.matches.clear()
            game.swap_players()

        data_manager.write()


def main():
    TicTacToe.keep_dense_learning()
    # TicTacToe.create_automated_game('static', 'random', num_matches=10)


if __name__ == '__main__':
    main()
