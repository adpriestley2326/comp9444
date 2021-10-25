"""
   kuzu.py
   COMP9444, CSE, UNSW
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class NetLin(nn.Module):
    # linear function followed by log_softmax
    def __init__(self):
        super(NetLin, self).__init__()
        self.fc1 = nn.Linear(28*28,10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.log_softmax(self.fc1(x), 0)
        return out

class NetFull(nn.Module):
    # two fully connected tanh layers followed by log softmax
    def __init__(self):
        super(NetFull, self).__init__()
        num_hidden = 100
        self.fc1 = nn.Linear(28*28, num_hidden)
        self.fc2 = nn.Linear(num_hidden, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.tanh(self.fc1(x))
        out = F.log_softmax(self.fc2(out), 0)
        return out

class NetConv(nn.Module):
    # two convolutional layers and one fully connected layer,
    # all using relu, followed by log_softmax
    def __init__(self):
        super(NetConv, self).__init__()
        # Parameters to play around with
        kernel1 = 5
        kernel2 = 5
        filters1 = 30
        filters2 = 20

        self.conv1 = nn.Conv2d(1, filters1, kernel1, padding=2)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel2)
        final_width = int(((32-kernel1+1)/2 - kernel2 + 1)/1)
        self.fc1 = nn.Linear(filters2*final_width*final_width, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        #out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.log_softmax(self.fc1(out), 0)
        return out
