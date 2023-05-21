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


toggle_action_space = action_space=spaces.Discrete(5)

SCALE = 100

def toggle_observation_preprocessor(obs: Observation, dyn:ModelDynamics):
    infected = SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])**(1/4)
    dead = SCALE * np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities])**(1/4)
    confined = np.ones_like(dead)*int((dyn.get_action()['confinement']))
    isolated = np.ones_like(dead)*int((dyn.get_action()['isolation']))
    beds_added = np.ones_like(dead)*int((dyn.get_action()['hospital']))
    vaccinated = np.ones_like(dead)*int((dyn.get_action()['vaccinate']))
    return torch.Tensor(np.stack((infected, dead, confined, isolated, beds_added,vaccinated))).unsqueeze(0)

ACTION_NULL = 0
ACTION_CONFINE = 1
ACTION_ISOLATE = 2
ACTION_HOSPITAL = 3
ACTION_VACCINATE = 4

def toggle_action_preprocessor(a:torch.Tensor, dyn:ModelDynamics):
    action = dyn.get_action()

    if a == ACTION_CONFINE:
        action['confinement'] = not action['confinement']
    elif a == ACTION_ISOLATE:
        action['isolation'] = not action['isolation']
    elif a == ACTION_VACCINATE:
        action['vaccinate'] = not action['vaccinate']
    elif a == ACTION_HOSPITAL:
        action['hospital'] = not action['hospital']
        
    return action