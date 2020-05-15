from q_learn.q_learn import QLearn
from nim.nim import Nim
import numpy as np
from mcts.mcts import Mcts
from mcts.pd_logger import PDLogger
from datetime import datetime
from pathlib import Path
from nim.random_agent import RandomAgent
from nim.expert_agent import ExpertAgent
import random


def print_q(ql: QLearn):
    for state in ql.q_table:
        print(f'{state}: ')
        for i, action in enumerate(ql.possible_actions[state]):
            print(f'{action}: {ql.q_table[state][i]}')


def print_rewards(rew, num_of_ep):
    rew = np.split(np.array(rew), num_of_ep / 1000)
    count = 1000

    for r in rew:
        print(count, ": ", str(sum(r / 1000)))
        count += 1000


print("Hello in Nim")
piles, objects = input(
    "Set game settings (`number of piles` `number of objects`): ").split()

game = Nim(int(piles), int(objects))
QL = QLearn(game)
state_copy = game.get_state()

QL.train(ExpertAgent())
print_rewards(QL.reward_all_ep, 100000)
print()

count = 0
for i in range(10):
    print('Try yourself against QL :)')
    game.set_state(state_copy, False, 0)

    tree = Mcts(game, exploration_parameter=1.41)
    tree.run(1)

    winner = 0

    while not game.done:
        print(game.piles)
        action = QL.select_move(game)
        print('CPU 0 move: %s' % str(action))
        game.act(action)
        tree.move_root(action)

        if game.done:
            winner = 1
            break

        print(game.piles)
        tree.run(1200)
        action = tree.predict()
        game.act(action)
        tree.move_root(action)


        # move = input("Your move (`pile` `objects`): ").split()
        # action = tuple(int(x) for x in move)
        #
        # game.act(action)

    if winner == 1:
        print("QL won!")
    else:
        print("Player won!")
    count += 1 if winner == 1 else 0


print(f'QL won {count} out of 10')