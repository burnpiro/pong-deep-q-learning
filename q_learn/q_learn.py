from __future__ import annotations
from typing import Union
from typing import List, Set, Dict, Tuple
import numpy as np
import random
from nim.nim import Nim, ACTION as NIM_ACTION
from pong.pong_game import PongGame, ACTION as PONG_ACTION

AVAILABLE_ACTION = Union[NIM_ACTION, PONG_ACTION]
GAME = Union[Nim, PongGame]

NUM_EP = 10000
MAX_STEPS = 100

LR = 0.1
DISCOUNT_RATE = 0.99

START_EXP_RATE = 1.0
MAX_EXP_RATE = 1.0
MIN_EXP_RATE = 0.01
EXP_RATE_DECAY = 0.001


class QLearn:
    def __init__(self, possible_actions: List[AVAILABLE_ACTION], game: GAME, exp_rate: float = START_EXP_RATE,
                 min_exp_rate: float = MIN_EXP_RATE,
                 exp_rate_decay: float = EXP_RATE_DECAY, max_steps: int = MAX_STEPS, num_ep: int = NUM_EP,
                 lr: float = LR, discount_rate: float = DISCOUNT_RATE):
        self.game = game
        self.possible_actions = possible_actions
        self.exp_rate = exp_rate
        self.min_exp_rate = min_exp_rate
        self.exp_rate_decay = exp_rate_decay
        self.num_of_episodes = num_ep
        self.max_steps = max_steps
        self.lr = lr
        self.discount_rate = discount_rate
        self.q_table = np.zeros((game.get_action_space_size(), len(possible_actions)))
        self.reward_all_ep = []

    def __call__(self, *args, **kwargs):
        for episodes in range(self.num_of_episodes):
            done = False
            state = 0
            curr_reward = 0

            for step in range(self.max_steps):
                exp_rate_threshold = random.uniform(0, 1)
            if exp_rate_threshold > self.exp_rate:
                action = np.argmax(self.q_table[state, :])
            else:
                action = self.game.possible_actions().sample()

            done = self.game.act(action)

            self.q_table[state, action] = self.q_table[state, action] * (1 - self.lr) + self.lr * (
                        reward + self.discount_rate * np.map(self.q_table[new_state, :]))
