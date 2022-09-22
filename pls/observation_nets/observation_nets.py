import torch.nn as nn
import torch as th
import torch.nn.functional as F

class Observation_net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Observation_net, self).__init__()
        self.hidden = nn.Linear(input_size, 128)
        self.output = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = th.flatten(x, 1)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        # s = self.sigmoid(x)
        return x


class Observation_net_cnn(nn.Module):
    def __init__(self, input_size, output_size):
        super(Observation_net_cnn, self).__init__()
        # input (1, 49, 49)
        # convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1) # (8, 49,, 49) # pool -> (8, 24, 24)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # (16, 24, 24) # pool -> (16, 12, 12)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # (32, 12, 12) # pool -> (32, 6, 6)
        # linear layers
        # self.fc1 = nn.Linear(1152, 64) # downsampling_size = 10
        self.fc1 = nn.Linear(1568, 64) # downsampling_size = 8
        # self.fc1 = nn.Linear(3200, 64) # downsampling_size = 6
        self.fc2 = nn.Linear(64, output_size)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)

        # self.extractor_network = nn.Sequential(  # Input shape (482, 482)
        #     nn.Conv2d(1, 8, kernel_size=3, stride=1),  # ()
        #     nn.ReLU(),
        #     nn.Conv2d(8, 16, kernel_size=3, stride=1),  # ()
        #     nn.ReLU(),
        #     nn.Linear()
        #     nn.ReLU()
        # )

    def forward(self, x:th.Tensor):
        # convolutional layers with ReLU and pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flattening the image
        # x = x.view(-1, 6*6*32) # downsampling_size = 10
        x = x.view(-1, 1568) # downsampling_size = 8
        # x = x.view(-1, 3200) # downsampling_size = 6
        # linear layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
