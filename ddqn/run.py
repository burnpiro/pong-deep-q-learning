from .DDQN import DQN
from .wrappers import normalize_obs, reward_wrapper
import numpy as np
import random
import gym
import sys
import time
import os
from gym.envs.atari import AtariEnv
from gym.wrappers import Monitor
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

RAM_BALL_X_POS = 49

if __name__ == '__main__':
    env = AtariEnv(frameskip=1)
    # env = Monitor(env, '.', force=True)
    # env = reward_wrapper(env)
    env = normalize_obs(env)
    dqn = DQN(env.observation_space.shape[0], gym.spaces.Discrete(4))
    dir_ = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(dir_, 'model2')
    print(model_file)
    dqn.model.load_weights(model_file)
    dqn.epsilon = 0.0
    ball_near_player = False

    for i in range(1):
        scores = [0, 0]
        prev_bounce = 0

        state = env.reset()
        done = False

        rewards_sum = 0
        avg_loss = 0

        step = 0
        player1_bounces = 0
        while not done:
            action = dqn.act(state)
            if state[RAM_BALL_X_POS] > 0:
                ball_near_player = True

            next_state, reward, done, _ = env.step(action)
            if reward:
                ball_near_player = False

            if ball_near_player and next_state[RAM_BALL_X_POS] < 0:
                ball_near_player = False
                player1_bounces += 1
                print(player1_bounces)

            env.render()
            rewards_sum += reward

            state = next_state
            if reward:
                scores[0 if reward == 1 else 1] += 1
                print(scores)

            if prev_bounce != player1_bounces:
                prev_bounce = player1_bounces
            # time.sleep(0.1)

        print(i, rewards_sum, dqn.epsilon)

    print(player1_bounces)
    print(scores)
