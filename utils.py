
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


from statistics import median

SCALE = 100
ACTION_NULL = 0
ACTION_CONFINE = 1
ACTION_ISOLATE = 2
ACTION_HOSPITAL = 3
ACTION_VACCINATE = 4



def moving_median(data, window_size):
    medians = []
    for i in range(window_size, len(data) - window_size + 1):
        window = data[i-window_size:i+window_size]
        med = median(window)
        medians.append(med)
    return medians

def plot_training(training_traces, eval_traces,window_size = 30):
    colors = ['red', 'blue', 'green'] 
    x = list(range(len(training_traces[0]))) 
    _, ax = plt.subplots()

    for i, training_trace in enumerate(training_traces):
            y = training_trace 
            ax.scatter(x, y, color=colors[i], label=f'Training {i+1}')

    ax.legend() 
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Rewards')
    plt.show()
    moving_medians = [moving_median(training_trace, window_size) for training_trace in training_traces]

    _, ax = plt.subplots()
    x = list(range(window_size, len(x)- window_size + 1)) 
    for i, moving_med in enumerate(moving_medians):
            y = moving_med
            ax.plot(x,y, color=colors[i], label=f'Training {i+1}')

    ax.legend() 
    ax.set_xlabel('Training steps')
    ax.set_ylabel('Rewards')
    plt.show()

    x = list(range(0,500,50))
    avg_eval_trace = np.mean(np.array(eval_traces), axis=0)
    plt.plot(x,avg_eval_trace)
    plt.xlabel('Training steps')
    plt.ylabel('Average evaluation rewards')
    plt.show()
    return

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

def run_episode(agent : Agent, env : Env, seed = 0 ) : #runs a 30 week episode, of a given environment and with a given agent

    log = []
    rwds = []
    finished = False
    obs, info = env.reset(seed)
    while not finished:
        action = agent.act(obs)
        obs, rwd, finished, info = env.step(action)
        log.append(info)
        rwds.append(rwd.item())

    return log, rwds

def plot_episode(log, dyn, plot_actions = False) : 
   
    """ Parse the logs """
    total = {p:np.array([getattr(l.total,p) for l in log]) for p in dyn.parameters[:-1]} #we don't plot the total pop
    cities = {c:{p:np.array([getattr(l.city[c],p) for l in log]) for p in dyn.parameters[:-1]} for c in dyn.cities}
    

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

def extract_episode_features(log, rwds):
        
        N_confinement = 7 * sum([l.action['confinement'] for l in log])
        cumulative_rwd = sum(rwds)
        N_deaths = log[-1].total.dead
        
        return [N_confinement, cumulative_rwd, N_deaths]

def hist_avg(ax, data,title):
        ymax = 50
        if title == 'deaths':
            x_range = (1000,200000)
        elif title == 'cumulative rewards': 
            x_range = (-300,300)
        elif 'days' in title:
            x_range = (0,200)
        else:
            raise ValueError(f'{title} is not a valid title') 
        ax.set_title(title)
        ax.set_ylim(0,ymax)
        ax.vlines([np.mean(data)],0,ymax,color='red')
        ax.hist(data,bins=60,range=x_range)      
        
def plot_episodes_features(episodes_features):
    
    N_confinements = np.array(episodes_features['conf_days'])
    cumulative_rewards = np.array(episodes_features['cumulative_rwd'])
    N_deaths = np.array(episodes_features['deaths'])
    
    fig, ax = plt.subplots(3,1,figsize=(10,10))        

    hist_avg(ax[0], N_confinements,'confinement days')
    hist_avg(ax[1], cumulative_rewards,'cumulative rewards')
    hist_avg(ax[2], N_deaths,'deaths')
    fig.tight_layout()
    plt.show()

    """ Print example """
    print(f'Average death number: {np.mean(N_deaths)}')
    print(f'Average number of confined days: {np.mean(N_confinements)}')
    print(f'Average cumulative reward: {np.mean(cumulative_rewards)}')