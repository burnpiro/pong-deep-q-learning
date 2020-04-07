from mcts import Mcts
from pong.pong_game import PongGame
import pandas as pd
from time import sleep, time
from pong.gym_agents import *
from pd_logger import PDLogger
from datetime import datetime
from nim.nim import Nim
from pathlib import Path

print("Welcome in Nim")

playouts = []

# for piles in [2]:
#     for objects in [20, 40, 80]:
#         for run in [1200]:
#             for opponent in [1200]:
#                 for parameter in [1.0, 1.41, 3.0]:
#                     for i in range(0, 50):
#                         playouts.append({
#                             'runs': run,
#                             'piles': piles,
#                             'objects': objects,
#                             'opponent': opponent,
#                             'exploration_parameter': parameter
#                         })

# for piles in [3]:
#     for objects in [20, 40, 80]:
#         for run in [1200]:
#             for opponent in [1200]:
#                 for parameter in [1.0, 1.41, 3.0]:
#                     for i in range(0, 50):
#                         playouts.append({
#                             'runs': run,
#                             'piles': piles,
#                             'objects': objects,
#                             'opponent': opponent,
#                             'exploration_parameter': parameter
#                         })
#
#
for piles in [4]:
    for objects in [20, 40, 80]:
        for run in [1200]:
            for opponent in [1200]:
                for parameter in [1.0, 1.41, 3.0]:
                    for i in range(0, 50):
                        playouts.append({
                            'runs': run,
                            'piles': piles,
                            'objects': objects,
                            'opponent': opponent,
                            'exploration_parameter': parameter
                        })

for playout in playouts:
    print('Playing nim with {} runs, using {} piles with {} objects, EP = {}'.format(playout['runs'],
                                                                                     playout['piles'],
                                                                                     playout['objects'],
                                                                                     playout['exploration_parameter']))
    file_path = './logs-nim/' + str(playout['runs']) + \
                '-' + str(playout['opponent']) + \
                '-' + str(playout['exploration_parameter'])
    filename = file_path + '/nim-' + str(playout['piles']) + '-' + str(playout['objects']) + '-' + str(
        playout['runs']) + '_vs_' + str(playout['opponent']) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(filename)
    output_dir = Path(file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    game = Nim(int(piles), int(objects))

    nim_logger = PDLogger(filename)

    tree = Mcts(game, logger=nim_logger, exploration_parameter=playout['exploration_parameter'])
    tree.run(1)

    count = 0
    winner = 0

    while not game.done:
        count = count + 1
        tree.run(playout['runs'])
        action = tree.predict()
        game.act(action)
        tree.move_root(action)
        if game.done:
            print("You won!")
            winner = 1
            break

        tree.run(playout['opponent'])
        action = tree.predict()
        game.act(action)
        tree.move_root(action)

    nim_logger.save_to_file('p1' if winner == 1 else 'p2')
    # print(game.get_winner())
    # pong_logger.add_run_stats(pd.DataFrame({
    #     'winning_player': [game.get_winner()],
    #     'opponent': opponent_names[playout['agent']],
    #     'runs': [playout['runs']],
    #     'method': [playout['method']]
    # }))
