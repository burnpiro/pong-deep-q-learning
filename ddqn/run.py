from DDQN import DQN
from wrappers import partial_observation, stack_obs, reward_wrapper
import numpy as np
import random
import gym
import sys
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    env = gym.make('Pong-ram-v0')
    env = gym.wrappers.Monitor(env, '.', force=True)
    env = partial_observation(env, [60, 59, 54, 49, 18])
    env = stack_obs(env)
    env = reward_wrapper(env)
    dqn = DQN(env.observation_space.shape[0], env.action_space)
    dqn.model.load_weights("model2")
    dqn.epsilon = 0.0

    for i in range(100000):
        state = env.reset()
        state = state/255.0
        done = False

        rewards_sum = 0
        avg_loss = 0

        step = 0
        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            env.render()
            rewards_sum += reward

            next_state = next_state/255.0
            state = next_state
            # time.sleep(0.1)

        print(i, rewards_sum, dqn.epsilon)
