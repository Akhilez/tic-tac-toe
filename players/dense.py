import numpy as np
from NeuralNetworks.TicTacToe.players import Player
from NeuralNetworks.TicTacToe.framework.frame import Frame
from NeuralNetworks.TicTacToe.models.dense import DenseModel


class DenseNetworkPlayer(Player):

    TYPE = 'dense'

    def __init__(self, name, character=None):
        super().__init__(name, character)
        self.model = DenseModel(self.name)

    def get_positions(self, frame):
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        processed_frame = np.array([Frame.categorize_inputs(frame)]).reshape(1, 18)
        output = self.model.model.predict(processed_frame)[0]
        output = self.get_max(output, frame)
        return [int(output // 3), int(output % 3)]

    @staticmethod
    def get_max(output, frame):
        while True:
            max_index = output.argmax()
            indices = [max_index // 3, max_index % 3]
            if frame[indices[0]][indices[1]] is None:
                return max_index
            output[max_index] = -1
