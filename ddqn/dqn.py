import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
import gym
import sys
import time
from wrappers import partial_observation, stack_obs, reward_wrapper, normalize_obs
from gym.envs.atari import AtariEnv
from DDQN import DQN


if __name__ == '__main__':
    env = AtariEnv(frameskip=4)
    env = reward_wrapper(env)
    env = normalize_obs(env)
    # env = gym.make('LunarLander-v2')

    dqn = DQN(env.observation_space.shape[0], env.action_space)
    dqn.model.load_weights("model2")
    dqn.epsilon = 0.5
    ddqn_scores = []

    for i in range(100000000):
        state = env.reset()
        done = False

        rewards_sum = 0
        avg_loss = 0

        step = 0
        while not done:
            action = dqn.act(state)
            next_state, reward, done, _ = env.step(action)
            # env.render()

            rewards_sum += reward

            dqn.store_transition(
                state, action, float(reward), next_state, done)

            state = next_state
            step += 1
            if step % 10 == 0:
                dqn.train(batch_size=1024)

        ddqn_scores.append(rewards_sum)
        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])

        print(i, rewards_sum, avg_score, dqn.epsilon)
        if i % 10 == 0:
            dqn.model.save_weights('model2')
