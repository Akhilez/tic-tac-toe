import numpy as np
import torch
from players import Player


class PolicyGradPlayer(Player):

    TYPE = 'policy_grad'

    def __init__(self, name, character=None):
        super().__init__(name, character)

    def get_positions(self, frame):
        pass

    @staticmethod
    def get_character(character):
        return super().get_character(character)