import torch.nn as nn
import torch as th

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