import random 
import numpy as np
import torch 
from abc import ABC, abstractmethod, abstractproperty
from rlcard.agents.dqn_agent.typing import Transition

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

    @classmethod 
    def from_estimator_network(cls, estimator_network, memory_size, batch_size, sequence_length=None) -> "Memory":
        if estimator_network == 'mlp':
            return SimpleMemory(memory_size=memory_size, batch_size=batch_size)
        elif estimator_network == 'transformer':
            return RecurrentMemory(memory_size=memory_size, batch_size=batch_size, sequence_length=sequence_length)
        raise ValueError(f"Estimator network {estimator_network} not supported")


class SimpleMemory(Memory):
    ''' Memory for saving transitions
    '''
    def __init__(self, memory_size, batch_size):
        super().__init__(memory_size=memory_size, batch_size=batch_size)
        self._memory = []

    @property
    def memory(self):
        return self._memory
    
    @property
    def memory_size(self):
        return len(self._memory)

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
    ''' Memory for saving sequences of transitions
    '''
    def __init__(self, memory_size, batch_size, sequence_length):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
            sequence_length (int): the length of the sequence
        '''
        super().__init__(memory_size, batch_size)
        self.sequence_length = sequence_length
        self._memory = []

    @property
    def memory(self) -> list[Transition]:
        return self._memory

    @property 
    def memory_size(self):
        return len(self._memory)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of sequences of states
            action_batch (list): a batch of sequences of actions
            reward_batch (list): a batch of sequences of rewards
            next_state_batch (list): a batch of sequences of states
            done_batch (list): a batch of sequences of dones
        '''
        # Replicate the first transition if needed to pad the sequence size 
        memory = [self.memory[0].clone() for _ in range(self.sequence_length - 1)] + self.memory
        
        # Sample a batch of stanting indices of sequences 
        start_idx = torch.randint(0, self.memory_size, (self.batch_size, ))

        # 2. Get a batch of sequences 
        sequences = [memory[i:i+self.sequence_length] for i in start_idx] # [batch_size, sequence_length, 5]

        # sequence is a list of Transitions which are NamedTuples
        def unpack_and_cat_transitions(sequence):
            return (
                np.array([transition.state for transition in sequence]),
                sequence[-1].action, # Should be the last action in the sequence
                sequence[-1].reward, # TODO (Kacper) figure out if this should be the last reward or some combination of previous rewards
                np.array([transition.next_state for transition in sequence]), # This has to be a sequence 
                sequence[-1].done, # Sequence done if the last state done
                sequence[-1].legal_actions, # Take the last legal action
            )
        sequences = map(unpack_and_cat_transitions, sequences)
        samples = list(zip(*sequences))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)



# FIXME (Kacper) PaddedRecurrentMemory is still a work in progress, as a choice of padding Transition
# is not at all trivial. It cannot be simply something like -1s everywhere, and has to make some sense
# in the context of what transition encodings normally indicate.
class PaddedRecurrentMemory(Memory):

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
            Transition.empty_transition() for _ in range(sequence_length - 1)
        ]

    @property
    def memory(self):
        return self._memory

    @property 
    def memory_size(self):
        return len(self._memory) - self.sequence_length + 1

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of sequences of states
            action_batch (list): a batch of sequences of actions
            reward_batch (list): a batch of sequences of rewards
            next_state_batch (list): a batch of sequences of states
            done_batch (list): a batch of sequences of dones
        '''
        # 1. Sample a batch of starting indices of sequences 
        start_idx = torch.randint(0, self.memory_size, (self.batch_size, ))

        # 2. Get a batch of sequences 
        sequences = [self._memory[i:i+self.sequence_length] for i in start_idx] # [batch_size, sequence_length, 5]

        # sequence is a list of Transitions which are NamedTuples
        def unpack_and_cat_transitions(sequence):
            print(sequence)
            return (
                np.array([transition.state for transition in sequence]),
                np.array([transition.action for transition in sequence]),
                np.array([transition.reward for transition in sequence]),
                np.array([transition.next_state for transition in sequence]),
                np.array([transition.done for transition in sequence])
            )
        sequences = map(unpack_and_cat_transitions, sequences)
        samples = list(zip(*sequences))
        return tuple(map(np.array, samples[:-1])) + (samples[-1],)

