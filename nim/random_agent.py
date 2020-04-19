import random
from nim.agent import Agent


class RandomAgent(Agent):
    def select_mode(self, game):
        return random.choice(game.possible_actions())
