from __future__ import annotations
from typing import Union
from typing import List, Set, Dict, Tuple
import numpy as np
import random
import sys
from nim.agent import Agent
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
    def __init__(self, game: GAME, exp_rate: float = START_EXP_RATE,
                 min_exp_rate: float = MIN_EXP_RATE,
                 exp_rate_decay: float = EXP_RATE_DECAY, max_steps: int = MAX_STEPS, num_ep: int = NUM_EP,
                 lr: float = LR, discount_rate: float = DISCOUNT_RATE):
        self.game = game
        self.exp_rate = exp_rate
        self.min_exp_rate = min_exp_rate
        self.exp_rate_decay = exp_rate_decay
        self.num_of_episodes = num_ep
        self.max_steps = max_steps
        self.lr = lr
        self.discount_rate = discount_rate
        self.q_table = {}
        self.reward_all_ep = []

    def execute_action(self, action, possible_actions):
        action_number = possible_actions.index(action)
        done = self.game.act(action)
        new_state = self.game.get_state_name()
        reward = 0
        if done != 0:
            reward = done

        return action_number, done, new_state, reward

    def train(self, opponent: Agent):
        state_copy = self.game.get_state()
        for episode in range(self.num_of_episodes):
            self.game.set_state(state_copy, False, 0)
            state = self.game.get_state_name()
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.game.possible_actions()))
            curr_reward = 0

            for step in range(self.max_steps):
                exp_rate_threshold = random.uniform(0, 1)
                possible_actions = self.game.possible_actions()
                if exp_rate_threshold > self.exp_rate:
                    action_idx = np.argmax(self.q_table[state][:])
                    action = possible_actions[int(action_idx)]
                else:
                    action = random.choice(possible_actions)

                action_number, done, new_state, reward = self.execute_action(action, possible_actions)

                if new_state not in self.q_table:
                    self.q_table[new_state] = np.zeros(len(possible_actions))

                self.q_table[state][action_number] = self.q_table[state][action_number] * (1 - self.lr) + self.lr * (
                        reward + self.discount_rate * np.max(self.q_table[new_state]))

                state = new_state
                curr_reward += reward

                if done != 0:
                    break

                action = opponent.select_mode(self.game)
                possible_actions = self.game.possible_actions()
                action_number, done, new_state, reward = self.execute_action(action, possible_actions)

                if new_state not in self.q_table:
                    self.q_table[new_state] = np.zeros(len(possible_actions))
                state = new_state

                if done != 0:
                    break

            progress = (episode + 1) / self.num_of_episodes * 100
            sys.stdout.write('\r')
            sys.stdout.write('[%-20s] %d%%' % ('=' * int(progress / 5), int(progress)))
            sys.stdout.write('\r')
            sys.stdout.flush()

            self.exp_rate = self.min_exp_rate + (MAX_EXP_RATE - self.min_exp_rate) * np.exp(
                -self.exp_rate_decay * episode)
            self.reward_all_ep.append(curr_reward)
