
import matplotlib.pyplot as plt
from gym import spaces

"""Environment imports"""
from epidemic_env.env       import Env, Log
from epidemic_env.dynamics  import ModelDynamics, Observation
from epidemic_env.visualize import Visualize
from epidemic_env.agent     import Agent

"""Pytorch and numpy imports"""
import numpy as np
import torch
from torch import nn
from matplotlib.ticker import MultipleLocator

SCALE = 100
ACTION_NULL = 0
ACTION_CONFINE = 1
ACTION_ISOLATE = 2
ACTION_HOSPITAL = 3
ACTION_VACCINATE = 4


def action_preprocessor(a:torch.Tensor, dyn:ModelDynamics):
    action = { # DO NOTHING
        'confinement': False, 
        'isolation': False, 
        'hospital': False, 
        'vaccinate': False,
    }
    
    if a == ACTION_CONFINE:
        action['confinement'] = True
    elif a == ACTION_ISOLATE:
        action['isolation'] = True
    elif a == ACTION_VACCINATE:
        action['vaccinate'] = True
    elif a == ACTION_HOSPITAL:
        action['hospital'] = True
        
    return action
    
def observation_preprocessor(obs: Observation, dyn:ModelDynamics):
    infected = SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
    dead = SCALE * np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities])
    confined = np.ones_like(dead)*int((dyn.get_action()['confinement']))
    return torch.Tensor(np.stack((infected, dead, confined))).unsqueeze(0)

def run_episode(agent : Agent, env : Env, seed = 0 ) : #runs a 30 week episode, of a given environment and with a given agent

    log = []
    finished = False
    obs, info = env.reset(seed)
    while not finished:
        action = agent.act(obs)
        obs, R, finished, info = env.step(action)
        log.append(info)

    return log

def plot_episode(log, dyn, plot_actions = False) : 
   
    """ Parse the logs """
    total = {p:np.array([getattr(l.total,p) for l in log]) for p in dyn.parameters[:-1]} #we don't plot the total pop
    cities = {c:{p:np.array([getattr(l.city[c],p) for l in log]) for p in dyn.parameters} for c in dyn.cities}
    

    fig = plt.figure(figsize=(14,10))
    ax_leftstate = plt.subplot2grid(shape=(9, 2), loc=(0, 0), rowspan=4)
    ax_leftobs = plt.subplot2grid(shape=(9, 2), loc=(4, 0), rowspan=3)
    ax_right = [plt.subplot2grid(shape=(9, 2), loc=(0, 1), colspan=1)]
    ax_right += [plt.subplot2grid(shape=(9, 2), loc=(i, 1), colspan=1) for i in range(1,9)]
    ax_right = {k:ax_right[_id] for _id,k in enumerate(cities.keys())}

    #total full state plot
    [ax_leftstate.plot(y) for y in total.values()] 
    ax_leftstate.legend([rf'${k[0]}_{{total}}^{{[w]}}$' for k in total.keys()]) 
    ax_leftstate.set_title('Full state')
    ax_leftstate.set_ylabel('number of people in each state')

    #total observable state plot
    [ax_leftobs.plot(total[y]) for y in ['infected','dead']]
    ax_leftobs.legend([rf'${y}_{{total}}^{{[w]}}$' for y in ['i','d']])
    ax_leftobs.set_title('Observable state')
    ax_leftobs.set_ylabel('number of people in each state')

    
    #cities observable state plots
    [ax.plot(cities[c]['infected']) for c, ax in ax_right.items()]
    [ax.plot(cities[c]['dead']) for c, ax in ax_right.items()]
    [ax.set_ylabel(c) for c, ax in ax_right.items()]
    [ax.xaxis.set_major_locator(plt.NullLocator()) for c, ax in ax_right.items()]
    ax_right['Zürich'].set_xlabel('time $w$ (in weeks)')
    ax_right['Zürich'].xaxis.set_major_locator(MultipleLocator(2.000))
    ax_right['Lausanne'].legend([rf'${y}_{{city}}^{{[w]}}$' for y in ['i','d']])

    if plot_actions :
        actions = {a:np.array([l.action[a] for l in log]) for a in log[0].action.keys()}
        ax_leftactions = plt.subplot2grid(shape=(9, 2), loc=(7, 0), rowspan=3)
        ax_leftactions.imshow(np.array([v for v in actions.values()]).astype(np.uint8),aspect='auto')
        ax_leftactions.set_title('Actions')
        ax_leftactions.set_yticks([0,1,2,3])
        ax_leftactions.set_yticklabels(list(actions.keys()))
        ax_leftactions.set_xlabel(rf'time $w$ (in weeks)')

    fig.tight_layout()
    plt.show()


class NoAgent(Agent): #the agent that take no action
    def __init__(self,  env:Env):
        self.env = env
        
    def load_model(self, savepath): pass

    def save_model(self, savepath): pass

    def optimize_model(self): return 0
    
    def reset(self,): pass
    
    def act(self, obs):
        return 0