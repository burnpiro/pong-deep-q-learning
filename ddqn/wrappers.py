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


class normalize_obs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    @staticmethod
    def convert_obs(obs):
        return (obs-127.5)/127.5

    def observation(self, obs):
        return normalize_obs.convert_obs(obs)


RAM_PLAYER_1_POS = 60
RAM_BALL_Y_POS = 54
BOUNCE_COUNT = 17
RAM_BALL_X_POS = 49


class reward_wrapper(gym.Env):
    def __init__(self, env, bounce_coeff=0.005):
        super().__init__()
        self.env = env
        self.prev_state = None
        self.obs = None
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Discrete(4)
        self.bounce_coeff = bounce_coeff

    def reward(self, rew):
        state = self.obs
        prev_state = self.prev_state

        diff = 0
        if state[RAM_BALL_X_POS] > 155:
            diff = state[BOUNCE_COUNT] - prev_state[BOUNCE_COUNT]

        return diff*self.bounce_coeff + rew

    def reset(self):
        self.obs = self.env.reset()
        self.prev_state = self.obs
        return self.obs

    def step(self, action):
        self.prev_state = self.obs
        self.obs, reward, done, _ = self.env.step(action)
        return self.obs, self.reward(reward), done, {}

    def render(self, mode='human'):
        return self.env.render(mode)


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
