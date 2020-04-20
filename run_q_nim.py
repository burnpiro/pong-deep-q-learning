from q_learn.q_learn import QLearn
from nim.nim import Nim
import numpy as np
from nim.random_agent import RandomAgent


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

QL.train(RandomAgent())

print_rewards(QL.reward_all_ep, QL.num_of_episodes)

old_QL = QL
game.set_state(state_copy, False, 0)
QL = QLearn(game)
QL.train(old_QL)

print_rewards(QL.reward_all_ep, QL.num_of_episodes)

# for id, item in QL.q_table.items():
#     print(id, item)

while True:
    print('Try yourself against QL :)')
    game.set_state(state_copy, False, 0)

    while not game.done:
        print(game.piles)
        action = QL.select_move(game)
        print('CPU 0 move: %s' % str(action))
        game.act(action)

        if game.done:
            print("You lost!")
            exit()

        print(game.piles)
        move = input("Your move (`pile` `objects`): ").split()
        action = tuple(int(x) for x in move)

        game.act(action)

    print("You won!")
