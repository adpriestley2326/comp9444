"""
   frac.py
   COMP9444, CSE, UNSW
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Full2Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full2Net, self).__init__()
        self.fc1 = nn.Linear(2, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.fc1(input))
        self.hid2 = torch.tanh(self.fc2(self.hid1))
        out = torch.sigmoid(self.fc3(self.hid2))
        return out

class Full3Net(torch.nn.Module):
    def __init__(self, hid):
        super(Full3Net, self).__init__()
        self.fc1 = nn.Linear(2, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.fc3 = nn.Linear(hid, hid)
        self.fc4 = nn.Linear(hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.fc1(input))
        self.hid2 = torch.tanh(self.fc2(self.hid1))
        self.hid3 = torch.tanh(self.fc3(self.hid2))
        out = torch.sigmoid(self.fc4(self.hid3))
        return out

class DenseNet(torch.nn.Module):
    def __init__(self, num_hid):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(2, num_hid)
        self.fc2 = nn.Linear(2 + num_hid, num_hid)
        self.fc3 = nn.Linear(2 + num_hid + num_hid, 1)

    def forward(self, input):
        self.hid1 = torch.tanh(self.fc1(input))
        self.hid2 = torch.tanh(self.fc2(torch.cat((input, self.hid1),1)))
        out = torch.sigmoid(self.fc3(torch.cat((input, self.hid1, self.hid2),1)))
        return out
