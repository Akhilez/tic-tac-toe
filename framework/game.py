from framework.match import Match


class Game:

    def __init__(self, player1, player2):
        """
        :param player1: Player 1 (human|random|dense)
        :param player2: Player 2 (human|random|dense)
        """
        self.player_1, self.player_2 = player1, player2
        self.matches = []
        self.current_match = None
        self.num_matches = 0

    def start(self, epochs=None):
        while epochs is None or epochs > 0:
            match = Match(self.player_1, self.player_2, self.num_matches)
            self.current_match = match

            match.start()
            self.num_matches += 1

            self.print_scores()
            match_summary = match.summary()
            self.matches.append(match_summary)

            if epochs is None:
                if not self.choose_to_replay():
                    print("Closing the game. Bye!")
                    epochs = 0
            else:
                epochs -= 1

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
