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


class Observation_Net_Stars(nn.Module):
    def __init__(self, input_size, output_size):
        super(Observation_Net_Stars, self).__init__()
        # input (1, 61, 61)
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=1) # (8, 30, 30)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1) # (16, 14, 14)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1) # (32, 6, 6)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1) # (64, 2, 2)
        # linear layers
        self.fc1 = nn.Linear(256, output_size)
        self.fc2 = nn.Linear(256, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:th.Tensor):
        # convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # flattening the image
        x = x.view(-1, 256)
        # linear layers
        fires = self.fc1(x)
        coord = self.fc2(x)
        return th.cat((fires, coord), dim=1)

class Observation_Net_Sokoban(nn.Module):
    def __init__(self, input_size, output_size):
        super(Observation_Net_Sokoban, self).__init__()
        # input (1, 40, 40)
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=1) # (8, 19, 19)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1) # (16, 9, 9)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1) # (32, 4, 4)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1) # (64, 1, 1)
        # linear layers
        self.fc1 = nn.Linear(64, output_size)
        self.fc2 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:th.Tensor):
        # convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # flattening the image
        x = x.view(-1, 64)
        # linear layers
        fires = self.fc1(x)
        coord = self.fc2(x)
        return th.cat((fires, coord), dim=1)


class Observation_Net_Carracing(nn.Module):
    def __init__(self, input_size, output_size):
        super(Observation_Net_Carracing, self).__init__()
        # input (1, 40, 40)
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=1) # (8, 19, 19)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1) # (16, 9, 9)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=1) # (32, 4, 4)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1) # (64, 1, 1)
        # linear layers
        self.fc1 = nn.Linear(64, output_size)
        self.fc2 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x:th.Tensor):
        # convolutional layers with ReLU and pooling
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # flattening the image
        x = x.view(-1, 64)
        # linear layers
        fires = self.fc1(x)
        coord = self.fc2(x)
        return th.cat((fires, coord), dim=1)
