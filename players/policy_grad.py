import copy

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
        self.weights_1 = self.load_params('weights_1', shape=(56, 27))
        self.biases_1 = self.load_params('biases_1', shape=56)

        self.weights_2 = self.load_params('weights_2', shape=(9, 56))
        self.biases_2 = self.load_params('biases_2', shape=9)

    def get_positions(self, frame):
        if self.flip():
            return RandomPlayer.get_random_position(frame.matrix)
        frame = frame.matrix if self.character == Frame.X else Frame.flip(frame.matrix)
        frame_one_hot = self.get_one_hot_frame(frame)

        position_one_hot = self.forward(frame_one_hot)

        output = self.get_max_index(position_one_hot[0], frame)
        return [int(output // 3), int(output % 3)]

    def train(self, epochs, data_manager):
        for epoch in range(epochs):
            self.clear_grads()
            total_loss = 0
            for match in data_manager.data:
                if match['winner'] is not None:
                    x, y = self.get_mini_batch(match)
                    y_hat = self.forward(x)
                    loss = self.backward(y, y_hat)

                    total_loss += loss
            print(f'Loss = {total_loss}')

    def forward(self, x):
        self.clear_grads()
        h1 = x.mm(self.weights_1.T) + self.biases_1
        y_hat = torch.softmax(h1.mm(self.weights_2.T) + self.biases_2, dim=1)
        return y_hat

    def backward(self, y, y_hat):
        loss = torch.nn.functional.binary_cross_entropy(y_hat, y)
        loss.backward()

        learning_rates_1 = self.get_dynamic_learning_rates(len(self.weights_1.grad))
        learning_rates_2 = self.get_dynamic_learning_rates(len(self.weights_2.grad))

        self.weights_1 = self.weights_1 - learning_rates_1 * self.weights_1.grad
        self.biases_1 = self.biases_1 - learning_rates_1.T * self.biases_1.grad
        self.weights_2 = self.weights_2 - learning_rates_2 * self.weights_2.grad
        self.biases_2 = self.biases_2 - learning_rates_2.T * self.biases_2.grad

        self.clear_grads()
        return loss

    def get_mini_batch(self, match):
        """
        :param match: dict of inserts[current, position, frame], winner and id
        """
        x, y = [], []
        match = copy.deepcopy(match)
        inserts = match['inserts']
        winner_character = match['winner']
        winners_inserts = [insert for insert in inserts if insert['current'] == winner_character]
        if winner_character == Frame.O:
            for insert in winners_inserts:
                insert['frame'] = Frame.flip(insert['frame'])
        for insert in winners_inserts:
            x.append(self.get_one_hot_frame(insert['frame']).reshape(27))
            y.append(self.get_one_hot_position(insert['position'][0], insert['position'][1]))
        return torch.stack(x), torch.stack(y)

    def clear_grads(self):
        self.weights_1 = self.weights_1.detach().requires_grad_()
        self.biases_1 = self.biases_1.detach().requires_grad_()
        self.weights_2 = self.weights_2.detach().requires_grad_()
        self.biases_2 = self.biases_2.detach().requires_grad_()

    def load_params(self, name, shape):
        weights_1_path = f'data/{self.name}/{name}.pt'
        if os.path.exists(weights_1_path):
            return torch.load(weights_1_path)
        else:
            return self.get_new_weights(shape)

    def save_params(self):
        os.makedirs(f'data/{self.name}', exist_ok=True)
        torch.save(self.weights_1, f'data/{self.name}/weights_1.pt')
        torch.save(self.biases_1, f'data/{self.name}/biases_1.pt')
        torch.save(self.weights_2, f'data/{self.name}/weights_2.pt')
        torch.save(self.biases_2, f'data/{self.name}/biases_2.pt')

    @staticmethod
    def get_new_weights(shape):
        range_min = -0.5
        range_max = 0.5
        random_tensor = torch.rand(shape, requires_grad=True, dtype=torch.float32)
        scaled_tensor = (range_min - range_max) * random_tensor + range_max
        return scaled_tensor

    @staticmethod
    def get_dynamic_learning_rates(length, initial=0.1, decay=0.8):
        lrs = []
        lr = initial
        for i in range(length):
            lrs.append(lr)
            lr = lr * decay
        lrs.reverse()
        return torch.tensor([lrs]).T

    @staticmethod
    def flip():
        flipped = np.random.rand()
        return flipped > 0.8

    @staticmethod
    def get_one_hot_frame(frame):
        return torch.tensor(Frame.categorize_inputs(frame), dtype=torch.float32).reshape(1, 27)

    @staticmethod
    def get_one_hot_position(i, j):
        position = torch.zeros(9)
        position[i*3 + j] = 1
        return position

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
