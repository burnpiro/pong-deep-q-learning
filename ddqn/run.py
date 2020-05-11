from DDQN import DQN
from wrappers import normalize_obs, reward_wrapper
import numpy as np
import random
import gym
import sys
import time
import os
from gym.envs.atari import AtariEnv
from gym.wrappers import Monitor
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    env = AtariEnv(frameskip=1)
    env = Monitor(env, '.', force=True)
    env = reward_wrapper(env)
    env = normalize_obs(env)
    dqn = DQN(env.observation_space.shape[0], env.action_space)
    dqn.model.load_weights("model2")
    dqn.epsilon = 0.0

    for i in range(1):
        state = env.reset()
        done = False

        rewards_sum = 0
        avg_loss = 0

        step = 0
        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            rewards_sum += reward

            state = next_state
            # time.sleep(0.1)

        print(i, rewards_sum, dqn.epsilon)
