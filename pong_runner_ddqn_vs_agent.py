from mcts.mcts import Mcts
from pong.pong_game import PongGame
from time import time
from pong.gym_agents import *
from pong.monitor import PongMonitor
from ddqn.ddqn_agent import DdqnAgent

game = PongGame()
game = PongMonitor(game, ".", force=True)
game.reset()

opponent = AggressiveAgent()

ddqn_agent = DdqnAgent()

while not game.done:
    ob = game._get_obs()
    action1 = ddqn_agent.act(ob)
    game.act(action1)

    action2 = opponent.act(ob, player=1)
    game.act(action2)

    game.render()
