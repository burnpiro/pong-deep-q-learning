from gym import logger

from pong.pong_game import PongGame
from pong.gym_agents import *
from time import sleep

agent_agr = GreedyAgent
agent_gre = RandomAgent

if __name__ == '__main__':
    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = PongGame()
    env.seed(0)
    agent1 = agent_agr(env.action_space, player=1)
    agent2 = agent_gre(env.action_space, player=2)

    episode_count = 1
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action1 = agent1.act(ob, reward, done)
            action2 = agent2.act(ob, reward, done)
            ob, reward = env.step(action1, a2=action2)
            if reward != 0:
                print(reward)
            env.render()
            sleep(0.03)

    # Close the env and write monitor result info to disk
    env.close()
