import numpy as np
import torch
import os

from framework.frame import Frame
from players import Player
from players.random import RandomPlayer


class PolicyGradPlayer(Player):
    TYPE = 'policy_grad'

    def __init__(self, name, character=None):
        super().__init__(name, character)
        self.weights_1 = self.load_weights_1(shape=(9, 27))
        self.biases_1 = self.load_biases_1(shape=9)

    def get_positions(self, frame):
        if self.flip():
            return RandomPlayer.get_random_position(frame.matrix)
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        frame_one_hot = self.get_one_hot_frame(frame)

        position_one_hot = self.forward(frame_one_hot)

        output = self.get_max_index(position_one_hot[0], frame)
        return [int(output // 3), int(output % 3)]

    def train(self, epochs, data_manager):
        self.clear_grads()

    def forward(self, x):
        self.clear_grads()
        y_hat = torch.softmax(x.mm(self.weights_1) + self.biases_1, dim=1)
        return y_hat

    def backward(self, y, y_hat):
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        loss.backward()

        learning_rates = self.get_dynamic_learning_rates(len(self.weights_1.grad))
        self.weights_1 = self.weights_1 - learning_rates * self.weights_1.grad
        self.biases_1 = self.biases_1 - learning_rates.T * self.biases_1.grad

        self.clear_grads()

    def clear_grads(self):
        self.weights_1 = self.weights_1.detach().requries_grad_()
        self.biases_1 = self.biases_1.detach().requries_grad_()

    def load_weights_1(self, shape):
        weights_1_path = f'data/{self.name}/weights_1.pt'
        if os.path.exists(weights_1_path):
            return torch.load(weights_1_path)
        else:
            return self.get_new_weights(shape)

    def load_biases_1(self, shape):
        biases_path = f'data/{self.name}/biases_1.pt'
        if os.path.exists(biases_path):
            return torch.load(biases_path)
        else:
            return self.get_new_weights(shape)

    @staticmethod
    def get_new_weights(shape):
        range_min = -0.5
        range_max = 0.5
        random_tensor = torch.rand(shape, requires_grad=True)
        scaled_tensor = (range_min - range_max) * random_tensor + range_max
        return scaled_tensor

    @staticmethod
    def get_dynamic_learning_rates(length, initial=0.1, decay=0.9):
        lrs = []
        lr = initial
        for i in range(length):
            lrs.append(lr)
            lr = lr * decay
        lrs.reverse()
        return torch.tensor([lrs]).T

    @staticmethod
    def flip():
        return bool(np.random.randint(2))

    @staticmethod
    def get_one_hot_frame(frame):
        return torch.tensor(Frame.categorize_inputs(frame)).reshape(1, 27)

    @staticmethod
    def get_max_index(output, frame):
        i = 0
        while i < len(output):
            i += 1
            max_index = output.argmax()
            indices = [max_index // 3, max_index % 3]
            if frame[indices[0]][indices[1]] is None:
                return max_index
            output[max_index] = -1
        return -1
