import re

from NeuralNetworks.TicTacToe.players import Player


class HumanPlayer(Player):
    TYPE = 'human'

    def get_positions(self, frame):
        while True:
            positions = input('Enter position in "x y" format: ').strip()
            inputs = re.match(r'([0-2])[\s,-]+([0-2])', positions)
            if inputs is not None:
                inputs = inputs.groups()
                inputs = int(inputs[0]), int(inputs[1])
                if frame.matrix[inputs[0]][inputs[1]] is not None:
                    print('That position already has a value.')
                    continue
                return inputs
            print("Wrong input.")
