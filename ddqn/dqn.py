import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import gym
import sys
import time
from wrappers import partial_observation, stack_obs, reward_wrapper

from DDQN import DQN


RAM_PLAYER_1_POS = 60
RAM_BALL_Y_POS = 54


if __name__ == '__main__':
    env = gym.make('Pong-ram-v0')
    env = partial_observation(env, [60, 59, 54, 49, 18])
    env = stack_obs(env)
    env = reward_wrapper(env)

    dqn = DQN(env.observation_space.shape[0], env.action_space)

    for i in range(100000):
        full_state = env.reset()
        state = full_state/255.0
        done = False

        rewards_sum = 0
        avg_loss = 0

        step = 0
        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            env.render()

            next_state = next_state/255.0
            rewards_sum += reward

            dqn.store_transition(
                state, action, float(reward), next_state, done)

            state = next_state
            step += 1
            if step % 100 == 0:
                dqn.train(batch_size=1024)
                dqn.update_target()

        print(i, rewards_sum, dqn.epsilon)
        dqn.model.save_weights('model2')
