from mcts.mcts import Mcts
from pong.pong_game import PongGame
from time import time
from pong.gym_agents import *
from pong.monitor import PongMonitor
from ddqn.ddqn_agent import DdqnAgent

game = PongGame()
game = PongMonitor(game, ".", force=True)
game.reset()

mcts_agent = GreedyAgent()
tree = Mcts(game, simulation_agent=mcts_agent)


ddqn_agent = DdqnAgent()

while not game.done:
    ob = game._get_obs()
    action1 = ddqn_agent.act(ob)
    game.act(action1)
    tree.move_root(action1)

    tree.run(30, verbose=True)
    action2 = tree.predict()
    game.act(action2)
    tree.move_root(action2)

    game.render()
