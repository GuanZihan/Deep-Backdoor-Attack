import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, (5, 5), 1, 0)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.dropout3 = nn.Dropout(0.1)

#         self.maxpool4 = nn.MaxPool2d((2, 2))
#         self.conv5 = nn.Conv2d(32, 64, (5, 5), 1, 0)
#         self.relu6 = nn.ReLU(inplace=True)
#         self.dropout7 = nn.Dropout(0.1)

#         self.maxpool5 = nn.MaxPool2d((2, 2))
#         self.flatten = nn.Flatten()
#         self.linear6 = nn.Linear(64 * 4 * 4, 512)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.dropout8 = nn.Dropout(0.1)
#         self.linear9 = nn.Linear(512, 10)

#     def forward(self, x):
#         for module in self.children():
#             x = module(x)
#         return x

# Build the neural network, expand on top of nn.Module
# class Net(nn.Module):
#   def __init__(self):

      

#       # call super class constructor

#       super(Net, self).__init__()

      

#       # specify fully-connected (fc) layer 1 - in 28*28, out 100

#       self.linear1 = nn.Linear(28*28, 100, bias=True) # the linearity W*x+b

#       self.relu1 = nn.ReLU(inplace=True) # the non-linearity 

#       self.linear2 = nn.Linear(100, 75, bias=True)

#       self.relu2 = nn.ReLU(inplace=True)

      

#       # specify fc layer 2 - in 75, out 50

#       self.linear3 = nn.Linear(75, 50, bias=True) # the linearity W*x+b

#       self.relu3 = nn.ReLU(inplace=True) # the non-linarity

      

#       # specify fc layer 3 - in 50, out 10

#       self.linear4 = nn.Linear(50, 10) # the linearity W*x+b

      

#       # add a softmax to the last layer

#       self.softmax = nn.Softmax(dim=1) # the softmax

    

#   # define network forward pass

#   def forward(self, images):

      

#       # reshape image pixels

#       x = images.view(-1, 28*28)

      

#       # define fc layer 1 forward pass

#       x = self.relu1(self.linear1(x))

      

#       # define fc layer 2 forward pass

#       x = self.relu2(self.linear2(x))

#       # define fc layer 3 forward pass

#       x = self.relu3(self.linear3(x))

#       # define layer 3 forward pass

#       # x = self.softmax(self.linear4(x))

      

#       # return forward pass result

#       return x
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    # define layers
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

    self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=10)

  # define forward function
  def forward(self, t):
    # conv 1
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # conv 2
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2)

    # fc1
    t = t.reshape(-1, 12*4*4)
    t = self.fc1(t)
    t = F.relu(t)

    # fc2
    t = self.fc2(t)
    t = F.relu(t)

    # output
    t = self.out(t)
    # don't need softmax here since we'll use cross-entropy as activation.

    return t