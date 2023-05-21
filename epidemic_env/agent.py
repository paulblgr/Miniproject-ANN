"""Implementation of the agent classes and associated RL algorithms.
"""
import torch
from abc import ABC, abstractmethod
from typing import Tuple
from epidemic_env.env import Env

class Agent(ABC):
    """Implements acting and learning. (Abstract class, for implementations see DQNAgent and NaiveAgent).

    Args:
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    @abstractmethod
    def __init__(self,  env, *args, **kwargs):
        """
        Args:
            env (_type_): the simulation environment
        """
        
    @abstractmethod
    def load_model(self, savepath:str):
        """Loads weights from a file.

        Args:
            savepath (str): path at which weights are saved.
        """
        
    @abstractmethod
    def save_model(self, savepath:str):
        """Saves weights to a specified path

        Args:
            savepath (str): the path
        """
        
    @abstractmethod
    def optimize_model(self)->float:
        """Perform one optimization step.

        Returns:
            float: the loss
        """
    
    @abstractmethod
    def reset():
        """Resets the agent's inner state
        """
        
    @abstractmethod 
    def act(self, obs:torch.Tensor)->Tuple[int, float]:
        """Selects an action based on an observation.

        Args:
            obs (torch.Tensor): an observation

        Returns:
            Tuple[int, float]: the selected action (as an int) and associated Q/V-value as a float
        """


class RussoAgent(Agent):
    def __init__(self,  env:Env,
                # Additionnal parameters to be added here
                max_infected:int=2e4, # maximum number of infected people before confinement
                confinement_period:int= 4, # confinement period in weeks
                ):
        """
        Example agent implementation. Just picks a random action at each time step.
        """
        self.env = env
        self.time_confined = 0
        self.population_infection_limit = max_infected
        self.confinement_period = confinement_period
                
    def load_model(self, savepath):
        # This is where one would define the routine for loading a pre-trained model
        pass

    def save_model(self, savepath):
        # This is where one would define the routine for saving the weights for a trained model
        pass

    def optimize_model(self):
        # This is where one would define the optimization step of an RL algorithm
        return 0
    
    def reset(self):
        # This should be called when the environment is reset
        self.time_confined = 0
    
    def act(self, obs):
        # this takes an observation and returns an action
        # the action space can be directly sampled from the env
        confined = (self.time_confined < self.confinement_period) and (self.time_confined > 0)   
        if confined:
            self.time_confined += 1
            return 1
          
        elif obs > self.population_infection_limit:
            self.time_confined = 1
            return 1
        
        self.time_confined = 0
        return 0
    
    
class NoAgent(Agent): #the agent that take no action
    def __init__(self,  env:Env):
        self.env = env
        
    def load_model(self, savepath): pass

    def save_model(self, savepath): pass

    def optimize_model(self): return 0
    
    def reset(self,): pass
    
    def act(self, obs):
        return 0
    