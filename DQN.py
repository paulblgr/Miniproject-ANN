import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import numpy as np

from epidemic_env.env       import Env, Log
from epidemic_env.dynamics  import ModelDynamics, Observation
from epidemic_env.visualize import Visualize
from epidemic_env.agent     import Agent


class DQN(nn.Module): #Q network as shown in the Pytorch example

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.n_observations = n_observations
        self.hidden_layer1 = nn.Linear(n_observations, 64)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer3 = nn.Linear(32, 16)
        self.out_layer = nn.Linear(16, n_actions)

    def forward(self, x):
        #x has shape[batchsize, 3, 9 , 7] 
        x = F.relu(self.hidden_layer1(x.view(-1,self.n_observations)))
        x = F.relu(self.hidden_layer2(x))
        x = F.relu(self.hidden_layer3(x))
        return self.out_layer(x) #output [batchsize, n_actions]

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def observation_preprocessor_DQN(obs: Observation, dyn:ModelDynamics):
    infected = np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
    dead = np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities])
    confined = np.ones_like(dead)*int((dyn.get_action()['confinement']))
    return torch.Tensor(np.stack((infected, dead, confined))).unsqueeze(0)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent(Agent) :
    def __init__(self,  env: Env, eps = 0.7, lr = 5e-3 ):
        self.BATCH_SIZE = 2048
        self.BUFFER_SIZE = 20000
        self.GAMMA = 0.99
        #EPS_START = 0.9
        #EPS_END = 0.05
        #EPS_DECAY = 1000
        self.env = env
        self.eps = eps
        self.n_actions = env.action_space.n
        self.n_observations = np.prod(env.observation_space.shape)
        self.policy_net = DQN(self.n_observations,self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory =  ReplayMemory(self.BUFFER_SIZE)

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

    def optimize_model(self,   update_target : bool):
        """Perform one optimization step.

        Returns:
            float: the loss
        """
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE) #List[obs, action, reward, next_state]
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        if update_target :
            print("updated")
            self.target_net.load_state_dict(self.policy_net.state_dict())

    
    def reset():
        """Resets the agent's inner state
        """

    def add_memory(self, state, action, next_state, reward) :
        self.memory.push(state, action, next_state, reward)

    def act(self, obs:torch.Tensor):
        Q_vals  = self.policy_net(obs)
        sample = random.random()

        if sample <= 1 - self.eps:
            with torch.no_grad():
                return torch.argmax(Q_vals).view(1,1)
        else :
            return torch.argmin(Q_vals).view(1,1)
        """Selects an action based on an observation.

        Args:
            obs (torch.Tensor): an observation

        Returns:
            Tuple[int, float]: the selected action (as an int) and associated Q/V-value as a float
        """



def training_step(last_obs,env, DQNagent : DQNAgent, update_target : bool)  :
    action = DQNagent.act(last_obs)
    obs, rwd, finished, info = env.step(action.item)
    rwd_ten = torch.tensor([rwd])


    #if finished:
    #   next_obs = None
    #else:
    #    next_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # Store the transition in memory
    DQNagent.add_memory(last_obs, action, obs, rwd_ten)

    # Move to the next state
    #obs = next_obs

    # Perform one step of the optimization (on the policy network)
    DQNagent.optimize_model(update_target)

    return obs, rwd, finished, info  

def training_episode(env, DQNAgent, update_target : bool, seed = 0) :
    # Initialize the environment and get it's state
    log = []
    rwds = []
    obs, info = env.reset(seed)
    obs, rwd, finished, info = training_step(obs, env, DQNAgent, update_target)
    log.append(info)
    rwds.append(rwd)
    while not finished:
        obs, rwd, finished, info = training_step(obs, env, DQNAgent)
        log.append(info)
        rwds.append(rwd)

    return log, rwds