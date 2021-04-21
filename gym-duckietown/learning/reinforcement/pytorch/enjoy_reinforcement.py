import ast
import argparse
import logging

import os
import numpy as np

import time

from reinforcement.pytorch.utils import canny_lane

# Duckietown Specific
from reinforcement.pytorch.ddpg import DDPG
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper


def _enjoy():          
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env, canny=False) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = DDPG(state_dim, action_dim, max_action, net_type="cnn")
    policy.load(filename='ddpg', directory='reinforcement/pytorch/models/')

    obs = env.reset()
    done = False
    
    i = 0
    while True:
        while not done:
            action = policy.predict(np.array(obs))
            # Perform action
            obs, reward, done, _ = env.step(action)
            print(action, reward)
            env.render()
            time.sleep(0.1)
        print("\nRESET\n")
        done = False
        obs = env.reset()   
        np.save('env{}.npz'.format(i), obs)
        i += 1

if __name__ == '__main__':
    _enjoy()
