import numpy as np

from framework.frame import Frame
from models.convolutional import ConvolutionalModel
from players import Player
from players.dense import DenseNetworkPlayer


class ConvolutionalPlayer(Player):
    TYPE = 'convolutional'

    def __init__(self, name, character):
        super().__init__(name, character)
        self.model = ConvolutionalModel(name)

    def get_positions(self, frame):
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        processed_frame = np.array([Frame.categorize_inputs(frame)])
        output = self.model.model.predict(processed_frame)[0]
        output = DenseNetworkPlayer.get_max(output, frame)
        return [int(output // 3), int(output % 3)]
