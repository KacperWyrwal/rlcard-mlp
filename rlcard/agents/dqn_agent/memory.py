import random 
import numpy as np
import torch 
from rlcard.agents.dqn_agent.typing import Transition
from abc import ABC, abstractmethod, abstractproperty


class Memory(ABC):
    ''' Abstract class for memory
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.max_memory_size = memory_size
        self.batch_size = batch_size

    @abstractproperty
    def memory(self) -> list: 
        pass 

    @abstractproperty
    def memory_size(self) -> int:
        pass 

    def save(self, state, action, reward, next_state, legal_actions, done):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if self.memory_size == self.max_memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)

    def checkpoint_attributes(self):
        ''' Returns the attributes that need to be checkpointed
        '''
        
        return {
            'memory_size': self.memory_size,
            'batch_size': self.batch_size,
            'memory': self.memory
        }
            
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' 
        Restores the attributes from the checkpoint
        
        Args:
            checkpoint (dict): the checkpoint dictionary
            
        Returns:
            instance (Memory): the restored instance
        '''
        
        instance = cls(checkpoint['memory_size'], checkpoint['batch_size'])
        instance.memory = checkpoint['memory']
        return instance
    
    @abstractmethod
    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''

class SimpleMemory(Memory):
    ''' Memory for saving transitions
    '''
    def __init__(self, memory_size, batch_size):
        super().__init__(memory_size=memory_size, batch_size=batch_size)
        self._memory = []

    @property
    def memory(self):
        return self._memory

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        samples = tuple(zip(*samples))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)
    

class RecurrentMemory(Memory):

    # TODO (Kacper) This only works if the transitions are saved sequentially 
    # from each episode, and the episodes are saved sequentially too. 

    def __init__(self, memory_size, batch_size, sequence_length):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
            sequence_length (int): the length of the sequence
        '''
        super().__init__(memory_size, batch_size)
        self.sequence_length = sequence_length
        self._memory = [
            # TODO (Kacper) this should be initialised with (sequence_length - 1) of empty transitions 
            # To do this, we should decide how to sensibly implement empty transitions. Maybe another 
            # dimension or maybe take some inspiration from padding methods in language models. 
        ]

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        # 1. Temporarily 

        # 2. 


        samples = random.sample(self.memory, self.batch_size)
        samples = tuple(zip(*samples))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)

