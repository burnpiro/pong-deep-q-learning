import random
from nim.agent import Agent


class ExpertAgent(Agent):

    def __init__(self):
        super().__init__()
        self.possible_moves = []

    def select_move(self, game):
        return random.choice(game.possible_actions())
