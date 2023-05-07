import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DQN(nn.Module): #Q network as shown in the Pytorch example

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.hidden_layer1 = nn.Linear(n_observations, 64)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer3 = nn.Linear(32, 16)
        self.out_layer = nn.Linear(16, n_actions)

    def forward(self, x):
        x = F.relu(self.hidden_layer1(x))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        return self.out_layer(x)
    