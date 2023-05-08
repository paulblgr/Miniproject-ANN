import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

from epidemic_env.agent     import Agent

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
    
class DQNAgent(Agent) :
    def __init__(self,  env, eps = 0.7):
        self.env =env
        self.eps = 0.7
        self.Qnet = DQN(1,2)

    def load_model(self, savepath:str):
        """Loads weights from a file.

        Args:
            savepath (str): path at which weights are saved.
        """
        
    def save_model(self, savepath:str):
        """Saves weights to a specified path

        Args:
            savepath (str): the path
        """

    def optimize_model(self)->float:
        """Perform one optimization step.

        Returns:
            float: the loss
        """
    
    def reset():
        """Resets the agent's inner state
        """
        
    def act(self, obs:torch.Tensor)->Tuple[int, float]:
        Q_vals  = self.Qnet(obs)
        sample = random.random()

        if sample <= 1 - self.eps:
            with torch.no_grad():
                return torch.argmax(Q_vals)
        
        """Selects an action based on an observation.

        Args:
            obs (torch.Tensor): an observation

        Returns:
            Tuple[int, float]: the selected action (as an int) and associated Q/V-value as a float
        """
