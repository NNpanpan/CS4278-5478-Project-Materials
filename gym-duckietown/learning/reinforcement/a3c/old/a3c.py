import functools
import operator
import ast
import argparse
import logging
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import multiprocessing as mp
from collections import deque

from skimage.transform import rescale
from skimage.color import rgb2gray

# Duckietown Specific
from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ACTION_SPACE = [(0.75, 0.0), (0.5, 0.0), (0.1, 0.0), (0.1, 1.0), (0.1, -1.0)]

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def make_env():
    # Launch the env with our helper function
    env = launch_env()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env) # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    return env

class Agent(nn.Module):
    def __init__(self, action_dim=None, max_action=None):
        super(Agent, self).__init__()
    # ONLY TRUE IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.policy = nn.Linear(512, len(ACTION_SPACE))
        self.value = nn.Linear(512, 1)

        self.max_action = max_action

    def forward(self, x):

        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        policy = F.softmax(self.policy(x))
        value = self.value(x)

        return policy, value

    def reset(self):
        return

    def clone(self, num=1):
        return [ self.clone_one() for _ in range(num) ]

    def clone_one(self):
        return Agent(action_dim=None, max_action=None)



class Runner(object):
    def __init__(self, agent, ix, train=True, **kwargs):
        self.agent = agent
        self.train = train
        self.ix = ix
        self.reset = False
        self.states = []

        print ("---Runner #{}".format(ix))
        self.env = make_env()


    def get_value(self):
        """
        Returns just the current state's value.
        This is used when approximating the R.
        If the last step was
        not terminal, then we're substituting the "r"
        with V(s) - hence, we need a way to just
        get that V(s) without moving forward yet.
        """
        _input = self.preprocess(self.states)
        _, _, _, value = self.decide(_input)
        return value

    def run_episode(self, yield_every = 10, do_render = False):
        """
        The episode runner written in the generator style.
        This is meant to be used in a "for (...) in run_episode(...):" manner.
        Each value generated is a tuple of:
        step_ix: the current "step" number
        rewards: the list of rewards as received from the environment (without discounting yet)
        values: the list of V(s) values, as predicted by the "critic"
        policies: the list of policies as received from the "actor"
        actions: the list of actions as sampled based on policies
        terminal: whether we're in a "terminal" state
        """
        self.reset = False
        step_ix = 0

        rewards, values, policies, actions = [[], [], [], []]

        self.env.reset()

        # we're going to feed the last 4 frames to the neural network that acts as the "actor-critic" duo. We'll use the "deque" to efficiently drop too old frames always keeping its length at 4:
        states = deque([ ])

        # we're pre-populating the states deque by taking first 4 steps as "full throttle forward":
        while len(states) < 4:
            obs, r, _, _ = self.env.step((0.8, 0.0))
            # print(obs.shape)
            # state = self.env.render(mode='rgb_array')
            state = obs
            # print(state.shape)
            # state = self.env.render()
            states.append(state)
            logger.info('Init reward ' + str(r) )
        # exit(0)

        # we need to repeat the following as long as the game is not over yet:
        while True:
            # the frames need to be preprocessed (I'm explaining the reasons later in the article)
            _input = self.preprocess(states)

            # asking the neural network for the policy and value predictions:
            action, action_ix, policy, value = self.decide(_input, step_ix)

            # taking the step and receiving the reward along with info if the game is over:
            _, reward, terminal, _ = self.env.step(action)

            # explicitly rendering the scene (again, this will be explained later)
            state = self.env.render(mode='rgb_array')

            # update the last 4 states deque:
            states.append(state)
            while len(states) > 4:
                states.popleft()

            # if we've been asked to render into the window (e. g. to capture the video):
            if do_render:
                self.env.render()

            self.states = states
            step_ix += 1

            rewards.append(reward)
            values.append(value)
            policies.append(policy)
            actions.append(action_ix)

            # periodically save the state's screenshot along with the numerical values in an easy to read way:
            if self.ix == 2 and step_ix % 200 == 0:
                fname = './screens/duckie/screen-' + str(step_ix) + '-' + str(int(time.time())) + '.jpg'
                im = Image.fromarray(state)
                im.save(fname)
                state.tofile(fname + '.txt', sep=" ")
                _input.numpy().tofile(fname + '.input.txt', sep=" ")

            # if it's game over or we hit the "yield every" value, yield the values from this generator:
            if terminal or step_ix % yield_every == 0:
                yield step_ix, rewards, values, policies, actions, terminal
                rewards, values, policies, actions = [[], [], [], []]

            # following is a very tacky way to allow external using code to mark that it wants us to reset the environment, finishing the episode prematurely. (this would be hugely refactored in the production code but for the sake of playing with the algorithm itself, it's good enough):
            if self.reset:
                self.reset = False
                self.agent.reset()
                states = deque([ ])
                self.states = deque([ ])
                return

            if terminal:
                self.agent.reset()
                states = deque([ ])
                return

    def ask_reset(self):
        self.reset = True

    def preprocess(self, states):
        # return torch.stack([ torch.tensor(self.preprocess_one(image_data), dtype=torch.float32) for image_data in states ])
        return torch.stack([ torch.tensor(image_data, dtype=torch.float32) for image_data in states ])

    def preprocess_one(self, image):
        """
        Scales the rendered image and makes it grayscale
        """
        return rescale(rgb2gray(image), (0.24, 0.16), anti_aliasing=False, mode='edge', multichannel=False)

    def choose_action(self, policy, step_ix):
        """
        Chooses an action to take based on the policy and whether we're in the training mode or not. During training, it samples based on the probability values in the policy. During the evaluation, it takes the most probable action in a greedy way.
        """
        policies = ACTION_SPACE
        print(policy)

        if self.train:
            action_ix = np.random.choice(4, 1, p=torch.tensor(policy).detach().numpy())[0]
        else:
            action_ix = np.argmax(torch.tensor(policy).detach().numpy())

        logger.info('Step ' + str(step_ix) + ' Runner ' + str(self.ix) + ' Action ix: ' + str(action_ix) + ' From: ' + str(policy))

        return np.array(policies[action_ix], dtype=np.float32), action_ix

    def decide(self, state, step_ix = 999):
        policy, value = self.agent(state)

        action, action_ix = self.choose_action(policy, step_ix)

        return action, action_ix, policy, value

    def load_state_dict(self, state):
        """
        As we'll have multiple "worker" runners, they will need to be able to sync their agents' weights with the main agent.
        This function loads the weights into this runner's agent.
        """
        self.agent.load_state_dict(state)

    def predict(self, state):

        # just making sure the state has the correct format, otherwise the prediction doesn't work
        assert state.shape[0] == 3

        if self.flat:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        else:
            state = torch.FloatTensor(np.expand_dims(state, axis=0)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()