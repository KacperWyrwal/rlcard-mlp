from typing import Any

import torch 
from torch import nn 


class Estimator(object):
    '''
    Approximate clone of rlcard.agents.dqn_agent.Estimator that
    uses PyTorch instead of Tensorflow.  All methods input/output np.ndarray.

    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    '''

    def __init__(self, num_actions=2, learning_rate=0.001, state_shape=None, mlp_layers=None, device=None, estimator_network: str = 'mlp'):
        ''' Initilalize an Estimator object.

        Args:
            num_actions (int): the number output actions
            state_shape (list): the shape of the state space
            mlp_layers (list): size of outputs of mlp layers
            device (torch.device): whether to use cpu or gpu
        '''
        self.num_actions = num_actions
        self.learning_rate=learning_rate
        self.state_shape = state_shape
        self.mlp_layers = mlp_layers
        self.device = device

        # set up Q model and place it in eval mode
        if estimator_network == 'mlp':
            qnet = MLPEstimatorNetwork(num_actions, state_shape, mlp_layers)
        elif estimator_network == 'transformer':
            qnet = TransformerEstimatorNetwork(num_actions, state_shape)
        else:
            raise ValueError(f'Unknown estimator_network: {estimator_network}')
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()

        # initialize the weights using Xavier init
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        # set up loss function
        self.mse_loss = nn.MSELoss(reduction='mean')

        # set up optimizer
        self.optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        ''' Predicts action values, but prediction is not included
            in the computation graph.  It is used to predict optimal next
            actions in the Double-DQN algorithm.

        Args:
          s (np.ndarray): (batch, state_len)

        Returns:
          np.ndarray of shape (batch_size, NUM_VALID_ACTIONS) containing the estimated
          action values.
        '''
        with torch.no_grad():
            s = torch.from_numpy(s).float().to(self.device)
            q_as = self.qnet(s).cpu().numpy()
        return q_as

    def update(self, s, a, y):
        ''' Updates the estimator towards the given targets.
            In this case y is the target-network estimated
            value of the Q-network optimal actions, which
            is labeled y in Algorithm 1 of Minh et al. (2015)

        Args:
          s (np.ndarray): (batch, state_shape) state representation
          a (np.ndarray): (batch,) integer sampled actions
          y (np.ndarray): (batch,) value of optimal actions according to Q-target

        Returns:
          The calculated loss on the batch.
        '''
        self.optimizer.zero_grad()

        self.qnet.train()

        s = torch.from_numpy(s).float().to(self.device)
        a = torch.from_numpy(a).long().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        # (batch, state_shape) -> (batch, num_actions)
        # NOTE (Kacper) This indicates that the state representation should be a sequence for recurrent models
        q_as = self.qnet(s)

        # (batch, num_actions) -> (batch, )
        # NOTE (Kacper) Whereas this probably means that the action should be only the last action in the sequence
        Q = torch.gather(q_as, dim=-1, index=a.unsqueeze(-1)).squeeze(-1)

        # update model
        # NOTE (Kacper) this means that the reward should also be the reward of the last transition in the sequence 
        batch_loss = self.mse_loss(Q, y)
        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()

        self.qnet.eval()

        return batch_loss
    
    def checkpoint_attributes(self):
        ''' Return the attributes needed to restore the model from a checkpoint
        '''
        return {
            'qnet': self.qnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_actions': self.num_actions,
            'learning_rate': self.learning_rate,
            'state_shape': self.state_shape,
            'mlp_layers': self.mlp_layers,
            'device': self.device
        }
        
    @classmethod
    def from_checkpoint(cls, checkpoint):
        ''' Restore the model from a checkpoint
        '''
        estimator = cls(
            num_actions=checkpoint['num_actions'],
            learning_rate=checkpoint['learning_rate'],
            state_shape=checkpoint['state_shape'],
            mlp_layers=checkpoint['mlp_layers'],
            device=checkpoint['device']
        )
        
        estimator.qnet.load_state_dict(checkpoint['qnet'])
        estimator.optimizer.load_state_dict(checkpoint['optimizer'])
        return estimator
    

from abc import ABC
from torch import Tensor


class EstimatorNetwork(nn.Module, ABC):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions: int = 2, state_shape: torch.Size | None = None):
        ''' Initialize the Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
        '''
        super(EstimatorNetwork, self).__init__()

        self.num_actions = num_actions
        self.input_dims = math.prod(state_shape) if state_shape else 0

    def _validate_output(self, output: Tensor) -> None:
        if output.shape[-1] != self.num_actions: 
            raise RuntimeError((
                f'The last dimension of the output should be the number of actions. '
                f'Expected output.shape[-1] == {self.num_actions}, but got {output.shape[-1]}.'
            ))

    def __call__(self, *args: Any, **kwds: Any) -> Tensor:
        output = super().__call__(*args, **kwds)
        self._validate_output(output)
        return output
    

class MLPEstimatorNetwork(EstimatorNetwork):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, num_actions=2, state_shape=None, mlp_layers=None):
        ''' Initialize the Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(MLPEstimatorNetwork, self).__init__(num_actions=num_actions, state_shape=state_shape)

        # build the Q network
        layer_dims = [self.input_dims] + mlp_layers
        # NOTE (Kacper) I would personally ensure that the data is flat before input, instead of doing reshaping implicitly here. 
        # NOTE (Kacper) Also the batchnorm at input but not inbetween layers. Strange construction.
        layers = [nn.Flatten()]
        layers.append(nn.BatchNorm1d(layer_dims[0]))
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(layer_dims[-1], self.num_actions, bias=True)) # TODO (Kacper) Maybe we should add softmax after this lin?
        self._network = nn.Sequential(*layers)

    @property 
    def network(self) -> nn.Sequential:
        return self._network

    def forward(self, s) -> Tensor:
        return self.network(s)
    

import math 
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class AverageSequencePooling(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=self.dim)


class TransformerEstimatorNetwork(EstimatorNetwork):
    def __init__(self, num_actions=2, state_shape=None,  num_layers: int = 2, d_model: int = 128, nhead: int = 8, dim_feedforward: int = 32, dropout: float = 0.1):
        super().__init__(num_actions=num_actions, state_shape=state_shape)

        # TODO (Kacper) maybe we should add batchnorm before embedding as in the original MLP?
        # TODO (Kacper) also find out whether this embedding method with a simple linear layer is common
        embedding = nn.Linear(self.input_dims, d_model, bias=True)
        positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=5000)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout, 
            activation='relu', 
            layer_norm_eps=1e-5, 
            batch_first=True, # [batch, seq, feature]
            norm_first=False, # TODO (Kacper) check if modern version used layer norm prior to attention and feedforward or after
            bias=True, 
        )
        encoder = TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=None, # TODO (Kacper) check if modern architectures use layer norm (I don't think so)
            enable_nested_tensor=True,
        )
        pooling = AverageSequencePooling(dim=1) # 1 is the sequence dimension
        linear = nn.Linear(d_model, self.num_actions, bias=True)

        self._network = nn.Sequential(
            embedding,
            positional_encoding,
            encoder,
            pooling,
            linear, 
        )

    @property
    def network(self) -> TransformerEncoder:
        return self._network
    
    def forward(self, s) -> Tensor:
        return self.network(s)
