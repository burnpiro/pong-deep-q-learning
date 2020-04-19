from q_learn.q_learn import QLearn
from nim.nim import Nim
import numpy as np
from nim.random_agent import RandomAgent

print("Hello in Nim")
piles, objects = input(
    "Set game settings (`number of piles` `number of objects`): ").split()

game = Nim(int(piles), int(objects))
QL = QLearn(game)

QL.train(RandomAgent())

rewards = np.split(np.array(QL.reward_all_ep), QL.num_of_episodes / 100)
count = 100

for r in rewards:
    print(count, ": ", str(sum(r / 100)))
    count += 100

for id, item in QL.q_table.items():
    print(id, item)

# while not game.done:
#     print(game.piles)
#     # move = input("Your move (`pile` `objects`): ").split()
#     # action = tuple(int(x) for x in move)
#     tree.run(1200)
#     action = tree.predict()
#     print('CPU 0 move: %s' % str(action))
#     game.act(action)
#     tree.move_root(action)
#
#     if game.done:
#         print("You won!")
#         exit()
#
#     print(game.piles)
#
#     tree.run(1200)
#     action = tree.predict()
#     game.act(action)
#     tree.move_root(action)
#
#     if tree.root._state != game.piles:
#         print('!!!')
#         exit()
#
#     print("Enemy move: "+str(action))
#
# print("You lost!")
