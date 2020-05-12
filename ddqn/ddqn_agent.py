from .DDQN import DQN
import numpy as np
import random
import gym
import sys
import time
import os
from gym.envs.atari import AtariEnv
from gym.wrappers import Monitor
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

NOOP = 0
FIRE = 1
UP = 2
DOWN = 3


class DdqnAgent:
    def __init__(self):
        self.dqn = DQN(128, gym.spaces.Discrete(4))
        dir_ = os.path.dirname(os.path.abspath(__file__))
        model_file = os.path.join(dir_, 'model2')
        self.dqn.model.load_weights(model_file)
        self.dqn.epsilon = 0.0

    def observation(self, obs):
        return (obs-127.5)/127.5

    def act(self, observation, player=0):
        assert player == 0, 'DDQN agent works only for player 0'
        if observation is None:
            return FIRE

        state = self.observation(observation)
        action = self.dqn.act(state)
        return action
