import gym
import numpy as np


class partial_observation(gym.ObservationWrapper):
    def __init__(self, env, idx):
        super().__init__(env)
        self.idx = idx
        self.observation_space = gym.spaces.Box(0, 255, (len(idx),))

    def observation(self, obs):
        return obs[self.idx]


RAM_PLAYER_1_POS = 60
RAM_BALL_Y_POS = 54


class reward_wrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def reward(self, rew):
        state = self.env.obs[-2]
        next_state = self.env.obs[-1]
        pos = state[0]+5
        ball_pos = state[2]
        dist = abs(pos-ball_pos)

        next_pos = next_state[0]+5
        next_ball_pos = next_state[2]
        next_dist = abs(next_pos-next_ball_pos)
        return (dist-next_dist)*0.01 + rew


class stack_obs(gym.Env):
    def __init__(self, env, k=3):
        self.env = env
        self.observation_space = gym.spaces.Box(
            0, 255, (env.observation_space.shape[0]*k,))
        self.action_space = env.action_space
        self.k = k

    def reset(self):
        obs = self.env.reset()
        self.obs = np.stack([obs for _ in range(self.k)])
        return self.obs.flatten()

    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        self.obs[:-1] = self.obs[1:]
        self.obs[-1] = obs
        return self.obs.flatten(),  reward, done, {}

    def render(self, mode='human'):
        return self.env.render(mode)
