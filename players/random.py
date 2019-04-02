import random

from NeuralNetworks.TicTacToe.players import Player


class RandomPlayer(Player):
    TYPE = 'random'

    def get_positions(self, frame):
        return self.get_random_position(frame.matrix)

    @staticmethod
    def get_random_position(frame):
        positions = []
        for i in range(3):
            for j in range(3):
                if frame[i][j] is None:
                    positions.append((i, j))
        if len(positions) > 0:
            random_index = random.randint(0, len(positions) - 1)
            return positions[random_index]
