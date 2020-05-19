from mcts.mcts import Mcts
from pong.pong_game import PongGame
from time import time
from pong.gym_agents import *
from pong.monitor import PongMonitor
from ddqn.ddqn_agent import DdqnAgent

game = PongGame()
# game = PongMonitor(game, ".", force=True)

opponent = AggressiveAgent()
# opponent = GreedyAgent()
ddqn_agent = RandomAgent()

_scores = []
_bounces = []

for i in range(15):
    game.reset()
    scores = [0, 0]
    prev_bounce = 0
    while 21 not in scores:
        ob = game._get_obs()
        game.render()
        action1 = ddqn_agent.act(ob, player=0)
        game.act(action1)

        action2 = opponent.act(ob, player=1)
        reward = game.act(action2)

        # game.render()
        if reward:
            scores[0 if reward == 1 else 1] += 1

        if prev_bounce != game.bounce_count():
            prev_bounce = game.bounce_count()

    print(scores)
    print(game.bounce_count())
    _scores.append(scores[0]-scores[1])
    _bounces.append(prev_bounce)

print(f'average score: {sum(_scores)/len(_scores)}')
print(f'average bounces: {sum(_bounces)/len(_bounces)}')
