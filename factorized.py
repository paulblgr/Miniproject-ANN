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
from typing                 import Type

SCALE = 100

class DQN_factorized(nn.Module): #Q network as shown in the Pytorch example

    def __init__(self, n_observations, n_actions):
        super(DQN_factorized, self).__init__()
        self.n_observations = n_observations 
        self.hidden_layer1 = nn.Linear(n_observations, 64)
        self.hidden_layer2 = nn.Linear(64, 32)
        self.hidden_layer3 = nn.Linear(32, 16)
        self.out_layer = nn.Linear(16, n_actions*2)

    def forward(self, x):
        #x has shape[batchsize, obs_space, 9 , 7]
        x = x.view(-1,self.n_observations) #shape [batchsize,n_observations]
        x = F.relu(self.hidden_layer1(x)) #shape [batchsize, 64]
        x = F.relu(self.hidden_layer2(x)) #shape [batchsize, 32]
        x = F.relu(self.hidden_layer3(x)) #shape [batchsize, 16]
        return self.out_layer(x) #output [batchsize, n_actions*2]

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def DQN_observation_preprocessor_factorized(obs: Observation, dyn:ModelDynamics):
    infected = (SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities]))**(1/4)
    dead = (SCALE* np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities]))**(1/4)
    return torch.Tensor(np.stack((infected, dead))).unsqueeze(0)

def DQN_action_preprocessor_factorized(a:torch.Tensor, dyn:ModelDynamics):
    action = {
        'confine': bool(a[0]),
        'isolate': bool(a[1]),
        'hospital': bool(a[2]),
        'vaccinate': bool(a[3]),
    }
    
    return action

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


class DQN_factorizedAgent(Agent) :
    def __init__(self,  env: Env ,lr = 5e-3, C =5, BATCH_SIZE = 2480, BUFFER_SIZE = 20000, eps_0 = 0.7, gamma = 0.9 ,eps_min = None , Tmax = 500):
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
        self.policy_net = DQN_factorized(self.n_observations,self.n_actions)
        self.target_net = DQN_factorized(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory =  ReplayMemory(self.BUFFER_SIZE)

    def get_eps(self): 
        if self.eps_decay : return max(self.eps_0*(self.Tmax - self.t)/self.Tmax,self.eps_min)
        else : return self.eps

    def load_model(self, savepath:str):
        
        weights = torch.load(savepath)
        self.policy_net.load_state_dict(weights)
        self.target_net.load_state_dict(weights)
        
        
    def save_model(self, savepath:str):
       
        torch.save(self.policy_net.state_dict(), savepath + '.pth')

    def optimize_model(self):
        
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE) #List[(obs, action, reward, next_state)]
        
        batch = Transition(*zip(*transitions))

        next_state_batch = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_Qvalues = self.policy_net(state_batch).gather(1, action_batch) #shape[batch_size, 1]

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
        #torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1000)
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
        
        sample = torch.rand(1)

        if sample <= 1 - eps:
            with torch.no_grad():
                Q_vals = self.policy_net(obs)

            # Split the Q-values tensor into halves
            split_size = Q_vals.size(1) // 2
            Q_vals_split = torch.split(Q_vals, split_size, dim=1)

            # Choose the action for each decision independently
            action_batch = torch.cat([Q.max(1)[1].view(-1, 1) for Q in Q_vals_split], dim=1)

            # Select the overall action that maximizes the Q-values
            overall_action = action_batch.sum(dim=1)  # Sum the decisions to get overall action

            # Find the index of the overall action with the maximum Q-value
            overall_max_index = Q_vals[torch.arange(Q_vals.size(0)), overall_action].argmax()

            # Update the overall action based on the maximum index
            action_batch[overall_max_index] = overall_action[overall_max_index]

            return action_batch
        else:
            # Randomly sample each decision independently
            action_batch = torch.tensor([[self.env.action_space.sample()] for _ in range(obs.size(0))],
                                        dtype=torch.long)

            return action_batch



def training_step(last_obs,env, agent)  :
    action = agent.act(last_obs, False)
    
    print(action)    
    obs, rwd, finished, info = env.step(action.item())

    agent.add_memory(last_obs, action, obs, rwd)

    agent.optimize_model()

    return obs, rwd.item(), finished, info  

def training_episode(env, agent, seed) :
    log = []
    rwds = []
    obs, info = env.reset(seed)
    finished = False
    while not finished:
        obs, rwd, finished, info = training_step(obs, env, agent)
        log.append(info)
        rwds.append(rwd)

    return log, rwds

def training_loop(env, agent, first_seed, savepath : str ,Tmax = 500) :
    
    training_trace = []
    eval_trace  = []
    eval_env = copy.deepcopy(env)

    random.seed(first_seed * 100) 
    torch.manual_seed(first_seed * 100)
    torch.use_deterministic_algorithms(True)

    for i in tqdm(range(Tmax)):
        _, rwds = training_episode(env, agent, seed = first_seed + i)
        training_trace.append(np.array(rwds).sum())
        if i%50 == 0 or i == Tmax : 
            cumul_rwds = []
            for j in range(20) :
                _ ,rwds = run_episode(agent,eval_env, j)
                cumul_rwds.append(np.array(rwds).sum())
            
            avg_rwd = np.array(cumul_rwds).mean()

            if i == 0 : 
                agent.save_model(savepath)
            elif eval_trace[-1] < avg_rwd : agent.save_model(savepath) #we save only if the model is better
        
            eval_trace.append(avg_rwd)
        
    return training_trace, eval_trace