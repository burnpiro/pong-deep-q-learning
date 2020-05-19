from .DDQN import DQN
import numpy as np
import random
import gym
import sys
import time
import os
from gym.envs.atari import AtariEnv
from gym.wrappers import Monitor
from ddqn.wrappers import normalize_obs
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

NOOP = 0
FIRE = 1
UP = 2
DOWN = 5

dqn = DQN(128, gym.spaces.Discrete(4))
dir_ = os.path.dirname(os.path.abspath(__file__))
model_file = os.path.join(dir_, 'model2')
print('loading model from', model_file)
dqn.model.load_weights(model_file)
dqn.epsilon = 0.05
dqn.epsilon_min = 0.05


def dqn_heuristic(obs, player):
    actions = [NOOP, FIRE, UP, DOWN]

    if player == 0:
        scale = 1
    elif player == 1:
        scale = -1
    else:
        raise Exception(f'Wrong player action {player}')
    obs = normalize_obs.convert_obs(obs)
    return {action: scale*value for action, value in zip(actions, dqn.value(obs))}


class DdqnAgent:
    def __init__(self):
        pass

    def act(self, observation, player=0):
        assert player == 0, 'DDQN agent works only for player 0'
        if observation is None:
            return FIRE

        state = normalize_obs.convert_obs(observation)
        action = dqn.act(state)
        return action
