import random

import gym
import numpy as np
import torch
import cv2


def canny_lane(image):
    # Assume float img
    image = np.uint8(255 * image)
    # image = image.transpose(1, 2, 0)

    # Convert to grayscale here.
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(gray_image, 100, 200)
    # cannyed_image = np.float32(cannyed_image / 255)

    return cannyed_image

def find_lines(cannyed_image):

    lines = cv2.HoughLinesP(
        cannyed_image,
        rho=6,
        theta=np.pi / 200,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    return lines

def draw_lines(img, lines, color=[255, 0, 0], thickness=2, itype='gray'):
    # If there are no lines to draw, exit.
    if lines is None:
        return img
    
    # Make a copy of the original image.
    img = np.copy(img) 

    # Create a blank image that matches the original in size.
    if itype == 'gray':
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
            ),
            dtype=np.uint8,
        )    
    else:
        line_img = np.zeros(
            (
                img.shape[0],
                img.shape[1],
                3
            ),
            dtype=np.uint8,
        )    

    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)    
            
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)    

    # Return the modified image.
    return img


def seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# Code based on:
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

# Simple replay buffer
class ReplayBuffer(object):
    def __init__(self,max_size):
        self.storage = []
        self.max_size = max_size

    # Expects tuples of (state, next_state, action, reward, done)
    def add(self, state, next_state, action, reward, done):
        if len(self.storage) < self.max_size:
            self.storage.append((state, next_state, action, reward, done))
        else:
            # Remove random element in the memory beforea adding a new one
            self.storage.pop(random.randrange(len(self.storage)))
            self.storage.append((state, next_state, action, reward, done))


    def sample(self, batch_size=100, flat=True):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        states, next_states, actions, rewards, dones = [], [], [], [], []

        for i in ind:
            state, next_state, action, reward, done = self.storage[i]

            if flat:
                states.append(np.array(state, copy=False).flatten())
                next_states.append(np.array(next_state, copy=False).flatten())
            else:
                states.append(np.array(state, copy=False))
                next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        # state_sample, action_sample, next_state_sample, reward_sample, done_sample
        return {
            "state": np.stack(states),
            "next_state": np.stack(next_states),
            "action": np.stack(actions),
            "reward": np.stack(rewards).reshape(-1,1),
            "done": np.stack(dones).reshape(-1,1)
        }


def evaluate_policy(env, policy, eval_episodes=10, max_timesteps=500):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        step = 0
        while not done and step < max_timesteps:
            action = policy.predict(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            step += 1

    avg_reward /= eval_episodes

    return avg_reward
