import random
import functools
from nim.agent import Agent
from nim.nim import Nim


class ExpertAgent(Agent):

    def __init__(self):
        super().__init__()
        self.possible_moves = []

    def select_move(self, game: Nim):
        piles = game.get_state()
        nim_sum = functools.reduce(lambda x, y: x ^ y, piles)
        if nim_sum == 0:
            return random.choice(game.possible_actions())

        for index, pile in enumerate(piles):
            target_size = pile ^ nim_sum
            if target_size < pile:
                amount_to_move = pile - target_size
                return index, amount_to_move
