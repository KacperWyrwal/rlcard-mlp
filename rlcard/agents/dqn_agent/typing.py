from typing import NamedTuple


class Transition(NamedTuple):
    state: list[int] 
    action: int
    reward: float
    next_state: list[int]
    done: bool
    legal_actions: list[int]

    # TODO There are multiple choices for padding the sequence. One would be to designate a value for the 
    # state, action, reward, next_state, done, and legal_actions, which would indicate a padded transition. 
    # In this case the next_state would likely have to be provided as an argument.
    # Anothor option would be to replicate the first or the last transition.
    @classmethod 
    def empty_transition_leduc_poker(cls, next_state):
        raise NotImplementedError
        # return cls(
        #     state=..., 
        #     action=-1,
        #     reward=0., 
        #     next_state=next_state,
        #     done=False,
        #     legal_actions=...,
        # )
    
    def clone(self):
        return self.__class__(self.state.copy(), self.action, self.reward, self.next_state.copy(), self.done, self.legal_actions.copy())
    

