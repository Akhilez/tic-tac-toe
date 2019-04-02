from NeuralNetworks.TicTacToe.framework.match import Match


class Game:
    num_matches = 0

    def __init__(self, player1, player2):
        """
        :param player1: Player 1 (human|random|dense)
        :param player2: Player 2 (human|random|dense)
        """
        self.player_1, self.player_2 = player1, player2
        self.matches = []
        self.current_match = None

    def start(self, epocs=None):
        while epocs is None or epocs > 0:
            match = Match(self.player_1, self.player_2, Game.num_matches)
            self.current_match = match
            match.start()
            Game.num_matches += 1
            self.print_scores()
            match_summary = match.summary()
            self.matches.append(match_summary)
            if epocs is None:
                if self.choose_to_replay():
                    continue
                else:
                    print("Closing the game. Bye!")
                    epocs = 0
            else:
                epocs -= 1

    @staticmethod
    def choose_to_replay():
        choice = input("Replay? (y/n):").lower()
        return choice == 'y'

    def print_scores(self):
        print(f"Scores:\n\t{self.player_1}: {self.player_1.score}")
        print(f"\t{self.player_2}: {self.player_2.score}")

    def filter_draw_matches(self):
        return [match for match in self.matches if match.winner is not None]

    def swap_players(self):
        temp = self.player_1
        self.player_1 = self.player_2
        self.player_2 = temp
