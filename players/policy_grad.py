import numpy as np
import torch
import os
from players import Player


class PolicyGradPlayer(Player):
    TYPE = 'policy_grad'

    def __init__(self, name, character=None):
        super().__init__(name, character)
        self.weights_1 = self.load_weights_1(shape=(9, 27))
        self.biases_1 = self.load_biases_1(shape=9)

    def get_positions(self, frame):
        pass

    def train(self, epochs, data_manager):

        pass

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
