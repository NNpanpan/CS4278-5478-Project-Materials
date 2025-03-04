import gym
from gym import spaces
import numpy as np

from reinforcement.pytorch.utils import canny_lane, draw_lines, find_lines

class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, shape=(120, 160, 3)):
        super(ResizeWrapper, self).__init__(env)
        self.observation_space.shape = shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            shape,
            dtype=self.observation_space.dtype)
        self.shape = shape

    def observation(self, observation):
        from PIL import Image
        return np.array(Image.fromarray(observation).resize(self.shape[0:2]))


class NormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizeWrapper, self).__init__(env)
        self.obs_lo = self.observation_space.low[0, 0, 0]
        self.obs_hi = self.observation_space.high[0, 0, 0]
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, obs_shape, dtype=np.float32)

    def observation(self, obs):
        if self.obs_lo == 0.0 and self.obs_hi == 1.0:
            return obs
        else:
            return (obs - self.obs_lo) / (self.obs_hi - self.obs_lo)


class ImgWrapperCanny(gym.ObservationWrapper):
    def __init__(self, env=None, canny=True):
        super(ImgWrapper, self).__init__(env)
        self.canny = canny
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        canny_obs = canny_lane(observation)
        lines = find_lines(canny_obs)

        if self.canny:
            return np.array([draw_lines(canny_obs, lines)], np.float32)
        else:
            obs = np.uint8(observation * 255)
            obs = draw_lines(obs, lines, itype='rgb')
            return obs.transpose(2, 0, 1)

class ImgWrapper(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ImgWrapper, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class DtRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward == -1000:
            reward = -100
        # elif reward > 0:
        #     reward += 0
        # else:
        #     reward += 0

        return reward


# Deprecated
class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ActionWrapper, self).__init__(env)

    def action(self, action):
        action_ = [action[0], action[1]]
        return action_

class DiscreteWrapper_a6(gym.ActionWrapper):
    """
    Discrete actions (left, right, forward) instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(6)

    def action(self, action):
        speed = [0.4, 0.8]
        angle = [-1., 0., 1.]
        action_pool = [[speed[i], angle[j]] for i in range(len(speed)) for j in range(len(angle))]
        return np.array(action_pool[action])
