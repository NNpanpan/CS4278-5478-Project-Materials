import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent(nn.Module):
    def __init__(self, **kwargs):
        super(Agent, self).__init__(**kwargs)

        self.init_args = kwargs

        self.h = torch.zeros(1, 256)

        self.norm1 = nn.BatchNorm2d(4)
        self.norm2 = nn.BatchNorm2d(32)

        self.conv1 = nn.Conv2d(4, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(1152, 256)
        self.policy = nn.Linear(256, 4)
        self.value = nn.Linear(256, 1)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()

        nn.init.xavier_uniform_(self.conv1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv1.bias, 0.01)
        nn.init.xavier_uniform_(self.conv2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv2.bias, 0.01)
        nn.init.xavier_uniform_(self.conv3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv3.bias, 0.01)
        nn.init.xavier_uniform_(self.conv4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.conv4.bias, 0.01)
        nn.init.constant_(self.gru.bias_ih, 0)
        nn.init.constant_(self.gru.bias_hh, 0)
        nn.init.xavier_uniform_(self.policy.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.policy.bias, 0.01)
        nn.init.xavier_uniform_(self.value.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(self.value.bias, 0.01)

        self.train()

    def reset(self):
        self.h = torch.zeros(1, 256)

    def clone(self, num=1):
        return [ self.clone_one() for _ in range(num) ]

    def clone_one(self):
        return Agent(**self.init_args)

    def forward(self, state):
        state = state.view(1, 4, 96, 96)
        state = self.norm1(state)

        data = self.prelu1(self.conv1(state))
        data = self.prelu2(self.conv2(data))
        data = self.prelu3(self.conv3(data))
        data = self.prelu4(self.conv4(data))

        data = self.norm2(data)
        data = data.view(1, -1)

        h = self.gru(data, self.h)
        self.h = h.detach()

        pre_policy = h.view(-1)

        policy = F.softmax(self.policy(pre_policy))
        value = self.value(pre_policy)

        return policy, value