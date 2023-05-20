import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import numpy as np
import copy

from epidemic_env.env       import Env, Log
from epidemic_env.dynamics  import ModelDynamics, Observation
from epidemic_env.visualize import Visualize
from epidemic_env.agent     import Agent
from utils                  import run_episode
from tqdm                   import tqdm

SCALE = 100


class DQN(nn.Module): #Q network as shown in the Pytorch example

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.n_observations = n_observations #126
        self.hidden_layer1 = nn.Linear(n_observations, 64)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer3 = nn.Linear(32, 16)
        self.out_layer = nn.Linear(16, n_actions)

    def forward(self, x):
        #x has shape[batchsize, 2, 9 , 7]
        x = x.view(-1,self.n_observations) #shape [batchsize,n_observations = 126 ]
        x = F.relu(self.hidden_layer1(x)) #shape [batchsize, 64]
        x = F.relu(self.hidden_layer2(x)) #shape [batchsize, 32]
        x = F.relu(self.hidden_layer3(x)) #shape [batchsize, 16]
        return self.out_layer(x) #output [batchsize, n_actions]

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def observation_preprocessor_DQN(obs: Observation, dyn:ModelDynamics):
    infected = (SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities]))**(1/4)
    dead = (SCALE* np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities]))**(1/4)
    return torch.Tensor(np.stack((infected, dead))).unsqueeze(0)

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
    def __init__(self,  env: Env, lr = 5e-3, C =5, BATCH_SIZE = 2480, BUFFER_SIZE = 20000, eps_0 = 0.7, gamma = 0.9 ,eps_min = None , Tmax = 500):
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.GAMMA = gamma
        self.env = env
        self.t = 0 #time t
        self.C = C
        if eps_min != None : 
            self.eps_decay = True
            self.eps_0 = eps_0
            self.eps_min = eps_min
            self.Tmax = Tmax
            
        else : 
            self.eps_decay = False
            self.eps = eps_0

        self.n_actions = env.action_space.n
        self.n_observations = np.prod(env.observation_space.shape)
        self.policy_net = DQN(self.n_observations,self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory =  ReplayMemory(self.BUFFER_SIZE)

    def get_eps(self): 
        if self.eps_decay : return max(self.eps_0*(self.Tmax - self.t)/self.Tmax,self.eps_min)
        else : return self.eps

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
        torch.save(self.policy_net.state_dict(), savepath + '.pth')

    def optimize_model(self):
        """Perform one optimization step.

        Returns:
            float: the loss
        """
        if len(self.memory) < self.BATCH_SIZE:
            return 0
        transitions = self.memory.sample(self.BATCH_SIZE) #List[(obs, action, reward, next_state)]
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                    batch.next_state)), dtype=torch.bool)
        #non_final_next_states = torch.cat([s for s in batch.next_state
        #                                            if s is not None])
        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_Qvalues = self.policy_net(state_batch).gather(1, action_batch) #shape[batch_size, 1]

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_Qvalues = self.target_net(next_state_batch).max(1)[0]
        
        # Compute the expected Q values
        expected_state_action_Qvalues = (next_state_Qvalues * self.GAMMA) + reward_batch.squeeze()

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_Qvalues.squeeze(), expected_state_action_Qvalues)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1000)
        self.optimizer.step()

        if self.t%self.C == 0 :
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.t += 1

    
    def reset():
        """Resets the agent's inner state
        """

    def add_memory(self, state, action, next_state, reward) :
        self.memory.push(state, action, next_state, reward)

    def act(self, obs:torch.Tensor, eps_0 = True):
        if eps_0 : eps = 0 #no exploration
        else : eps = self.get_eps()
        
        sample = random.random()

        if sample <= 1 - eps:
            with torch.no_grad():
                Q_vals  = self.policy_net(obs)
            return Q_vals.max(1)[1].view(1,1)
        else :
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)



def training_step(last_obs,env, DQNagent : DQNAgent)  :
    action = DQNagent.act(last_obs, False)
    obs, rwd, finished, info = env.step(action.item())

    #if finished:
    #   obs = None
    #else:
    #    next_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # Store the transition in memory
    DQNagent.add_memory(last_obs, action, obs, rwd)

    # Perform one step of the optimization (on the policy network)
    DQNagent.optimize_model()

    return obs, rwd.item(), finished, info  

def training_episode(env, DQNAgent, seed) :
    log = []
    rwds = []
    obs, info = env.reset(seed)
    finished = False
    while not finished:
        obs, rwd, finished, info = training_step(obs, env, DQNAgent)
        log.append(info)
        rwds.append(rwd)

    return log, rwds

def training_loop(env, DQNagent, first_seed, savepath : str ,Tmax = 500) :
    
    training_trace = []
    eval_trace  = []
    eval_env = copy.deepcopy(env)

    random.seed(first_seed*100)
    np.random.seed(first_seed*100)
    torch.manual_seed(first_seed * 100)

    for i in tqdm(range(Tmax)):
        _, rwds = training_episode(env, DQNagent, seed = first_seed + i)
        training_trace.append(np.array(rwds).sum())
        if i%50 == 0 or i == Tmax : 
            cumul_rwds = []
            for j in range(20) :
                _ ,rwds = run_episode(DQNagent,eval_env, j)
                cumul_rwds.append(np.array(rwds).sum())
            
            if i == 0 : 
                DQNagent.save_model(savepath)

            avg_rwd = np.array(cumul_rwds).mean()
        
            eval_trace.append(avg_rwd)
        
    return training_trace, eval_trace