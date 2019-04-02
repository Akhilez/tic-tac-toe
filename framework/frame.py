import copy

from keras.utils import to_categorical


class Frame:
    X = 'X'
    O = 'O'

    win_lines = [
        [[0, 0], [1, 1], [2, 2]],  # [\]
        [[0, 2], [1, 1], [2, 0]],  # [/]
        [[0, 0], [1, 0], [2, 0]],  # [|  ]
        [[0, 1], [1, 1], [2, 1]],  # [ | ]
        [[0, 2], [1, 2], [2, 2]],  # [  |]
        [[0, 0], [0, 1], [0, 2]],  # [```]
        [[1, 0], [1, 1], [1, 2]],  # [---]
        [[2, 0], [2, 1], [2, 2]],  # [...]
    ]

    def __init__(self):
        self.matrix = self.generate_empty_canvas()

    def insert(self, player, row, column):
        self.matrix[row][column] = player.character

    def print_canvas(self):
        output = '\n\t0\t1\t2\n'
        for i in range(3):
            output += f'{i}\t'
            for j in range(3):
                value = self.matrix[i][j]
                value = value if value is not None else ' '
                output += f'{value}\t'
            output += '\n'
        output += '\n'
        print(output)

    def check_winner(self, player1, player2):
        for win_line in Frame.win_lines:
            num1 = self.matrix[win_line[0][0]][win_line[0][1]]
            num2 = self.matrix[win_line[1][0]][win_line[1][1]]
            num3 = self.matrix[win_line[2][0]][win_line[2][1]]

            if num1 is not None and num1 == num2 and num2 == num3:
                return player1 if player1.character == num1 else player2

    def is_canvas_filled(self):
        for row in self.matrix:
            for column in row:
                if column is None:
                    return False
        return True

    @staticmethod
    def generate_empty_canvas():
        return [
            [None, None, None],
            [None, None, None],
            [None, None, None]
        ]

    @staticmethod
    def flip(matrix):
        flipped = copy.deepcopy(matrix)
        for i in range(3):
            for j in range(3):
                if matrix[i][j] is None:
                    continue
                if matrix[i][j] == Frame.X:
                    flipped[i][j] = Frame.O
                else:
                    flipped[i][j] = Frame.X
        return flipped

    @staticmethod
    def categorize_inputs(my_list):
        categories = {None: [0, 0], 'X': [1, 0], 'O': [0, 1]}
        all_list = []
        for frame in my_list:
            category_list = []
            for position in frame:
                category_list.append(categories[position])
            all_list.append(category_list)
        return all_list

    @staticmethod
    def categorize_outputs(my_list):
        """
        :param my_list: outputs in the form: [1, 2]
        :return: Class of an output from 0-8, Ex: [1, 2] => 5
        """
        cat_list = []
        for lst in my_list:
            cat_list.append(lst[0] * 3 + lst[1])
        return to_categorical(cat_list, 9)
