from NeuralNetworks.TicTacToe.framework.frame import Frame
from NeuralNetworks.TicTacToe.framework.match import Match
from NeuralNetworks.TicTacToe.players import Player
from NeuralNetworks.TicTacToe.players.random import RandomPlayer


class PerfectPlayer(Player):

    TYPE = 'static'

    def get_positions(self, frame):
        """
        Conditions to get_position:
          - Check if you have opportunity
            - Place X where X has a win_line
          - Check if opponent has opportunity:
            - Place X where O has a win_line
          - Pick random position.
        """
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        position = Match.get_opportunity(frame, Frame.X)
        if position is None:
            position = Match.get_opportunity(frame, Frame.O)
            if position is None:
                position = Match.get_loose_opportunity(frame, Frame.X)
                if position is None:
                    position = RandomPlayer.get_random_position(frame)
        return position
