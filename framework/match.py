import copy
import random

from NeuralNetworks.TicTacToe.framework.frame import Frame


class Match:

    def __init__(self, player_1, player_2, match_id=None):
        self.frame = Frame()
        self.current_player = player_1
        self.other_player = player_2
        self.winner = None
        self.inserts = []
        self.id = match_id

    def start(self):
        print(f"Match ID: {self.id}")
        while True:
            self.frame.print_canvas()
            print(f'Current player = {self.current_player}')
            self.insert(self.current_player.get_positions(self.frame))
            winner = self.frame.check_winner(self.current_player, self.other_player)
            if winner is not None or self.frame.is_canvas_filled():
                self.frame.print_canvas()
                self.winner = winner
                self.print_winner(winner)
                self.update_scores(winner)
                return
            self.switch_players()

    def insert(self, positions):
        self.inserts.append({
            'current': self.current_player.character,
            'position': [positions[0], positions[1]],
            'frame': copy.deepcopy(self.frame.matrix)
        })
        self.frame.insert(self.current_player, positions[0], positions[1])

    @staticmethod
    def update_scores(winner):
        if winner is not None:
            winner.score += 1

    def summary(self):
        successful_inserts = self.get_best_inserts()
        # successful_inserts = self.remove_current_character_attribute(successful_inserts)
        return {
            'inserts': successful_inserts,
            'id': self.id,
            'winner': None if self.winner is None else self.winner.character}

    def get_best_inserts(self):
        """
        Criteria to decide which inserts were the best:
        - Remove opportunity given
        - Remove missed opportunities
        - Add winner's inserts
        """
        best_inserts = []
        for insert in self.inserts:
            frame = Frame.flip(insert['frame']) if insert['current'] == Frame.O else copy.deepcopy(insert['frame'])
            new_insert = copy.deepcopy(insert)
            new_insert['frame'] = frame
            if Match.is_best_position(frame, insert['position']):
                new_insert['best'] = True
            else:
                new_insert['best'] = False
            best_inserts.append(new_insert)
        return best_inserts

    @staticmethod
    def print_winner(winner):
        if winner is None:
            print('Draw!')
        else:
            print(f'{winner} won!')

    def switch_players(self):
        switcher = self.current_player
        self.current_player = self.other_player
        self.other_player = switcher

    @staticmethod
    def remove_current_character_attribute(inserts):
        for insert in inserts:
            del insert['current']
        return inserts

    @staticmethod
    def is_best_position(frame, current_position):
        """
        Steps:
          - X win opportunity:
            - Get opportunity position for X
            - if the position exists:
              - if position == current_position, then position makes sense. Return True.
          - O win opportunity:
            - Get opportunity position for O
            - if the position exists:
              - if position == current_position, then position makes sense, return True.
          - Return True.
        :param frame: X_Matrix - A frame matrix where the next player is always X
        :param current_position: Position that was selected for X
        :return: Returns True if the position made sense to win the match.
        """
        x_opportunity = Match.get_opportunity(frame, Frame.X)
        if x_opportunity is not None:
            return x_opportunity == current_position
        o_opportunity = Match.get_opportunity(frame, Frame.O)
        if o_opportunity is not None:
            return o_opportunity == current_position
        return True

    @staticmethod
    def get_opportunity(frame, character):
        for win_line in Frame.win_lines:
            num_chars = sum(frame[position[0]][position[1]] == character for position in win_line)
            if num_chars == 2:
                none_pos = [position for position in win_line if frame[position[0]][position[1]] is None]
                if len(none_pos) == 1:
                    return none_pos[0]

    @staticmethod
    def get_loose_opportunity(frame, character):
        for win_line in Frame.win_lines:
            num_chars = sum(frame[position[0]][position[1]] == character for position in win_line)
            if num_chars == 1:
                none_pos = [position for position in win_line if frame[position[0]][position[1]] is None]
                if len(none_pos) == 2:
                    return none_pos[random.randint(0, 1)]
