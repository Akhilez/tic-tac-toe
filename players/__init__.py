from abc import ABCMeta, abstractmethod

from NeuralNetworks.TicTacToe.framework.frame import Frame


class Player(metaclass=ABCMeta):
    TYPE = 'default'

    def __init__(self, name, character=None):
        self.name = name
        self.score = 0
        self.character = self.get_character(character)

    @abstractmethod
    def get_positions(self, frame):
        pass

    def get_character(self, character):
        if character is None:
            while True:
                character = input('Enter player 1 character (X or O): ').upper()
                if character == Frame.X or character == Frame.O:
                    break
                print(f'Please enter either {Frame.X} or {Frame.O}')
        return character

    def __str__(self):
        return f'{self.name} ({self.character})'

    def __eq__(self, other):
        return self.character == other.name
