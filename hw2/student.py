#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T


"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        #return T.ToTensor()
        return T.Compose([
            T.ToTensor(),
            T.RandomResizedCrop(size=80, scale=(0.8, 1.0)),
            #T.GaussianBlur(5),
            T.RandomAffine(60),
            T.ColorJitter(0.4, 0.4, 0.5, 0.2),
            T.RandomHorizontalFlip(),
            T.RandomAdjustSharpness(3),
            T.RandomErasing(),
            #T.RandomPerspective(),
        ])
    elif mode == 'test':
        return T.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 7, padding=3, padding_mode='reflect')
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv4 = nn.Conv2d(96, 96, 3, padding=1)
        self.conv5 = nn.Conv2d(96, 96, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.fc1 = nn.Linear(96*(10*10), 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128+128,8)
        #self.bn5 = nn.BatchNorm1d(32)
        #self.fc4 = nn.Linear(96+64+32,8)
        
    def forward(self, input):
        # Convolutional layers
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x,2)
        #x = F.dropout2d(x,p=0.2, training=self.training)
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.max_pool2d(x,2)
        #x = F.dropout2d(x,p=0.2, training=self.training)
        x = F.relu(self.conv3(x))
        #x = F.max_pool2d(x,2)
        #x = F.dropout2d(x, p=0.2, training=self.training)
        x = F.relu(self.conv4(x))
        x = F.relu(self.bn2(self.conv5(x)))
        x = F.max_pool2d(x,2)
        x = F.dropout2d(x, p=0.2, training=self.training)

        # Fully connected layers
        x = x.view(x.size(0), -1)
        x1 = F.relu(self.bn3(self.fc1(x)))
        #x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.relu(self.bn4(self.fc2(x1)) )
        #x2 = F.dropout(x2, p=0.2, training=self.training)
        out = F.log_softmax(self.fc3(torch.cat((x1,x2), dim=1)), dim=0)
        return out

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.01) # Kuzu used optim.SGD
optimizer = optim.AdamW(net.parameters(), lr=0.012, weight_decay=0.05)

loss_func = F.nll_loss # Kuzu used F.nll_los


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    return

scheduler = None

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 150
epochs = 200
