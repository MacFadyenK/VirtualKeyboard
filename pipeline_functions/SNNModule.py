# imports
import snntorch as snn
from snntorch import utils, surrogate

import torch
import torch.nn as nn
import torch.nn.functional as F

def createSNN(dim_inputs, hidden_layer, num_outputs=2, betas=[0.9, 0.9, 0.9], thresholds=[1, 1, 1], sigmoid_slope = 10, p=0.1):
    """
    Function wrapper that initiates a fully connected 3 layer SNN model
    
    Inputs:
    - dim_inputs: dimension size of the input features (flattened)
    - hidden_layer: number of neurons in the hidden layer of the model
    - num_outputs: number of output neurons/classes in the model. Default = 2
    - betas: decay constant for each LIF neuron layer as an array. Default = [0.9, 0.9]
    - thresholds: array of membrane potential thresholds for the neurons in each layer to produce a spike, Default = [1, 1]

    Returns: 
    - A fully connected 3 layer spiking neural network with the specified parameters
    """
    return fcSNN(dim_inputs=dim_inputs, hidden_layer=hidden_layer, num_outputs=num_outputs, betas=betas, thresholds=thresholds, sigmoid_slope=sigmoid_slope, p=p)


class fcSNN(nn.Module):
    def __init__(self, dim_inputs, hidden_layer, num_outputs, betas, thresholds, sigmoid_slope, p=0.1):
        super().__init__()

        # initializes lif and linear layers for the SNN
        self.fc1 = nn.Linear(dim_inputs, hidden_layer[0])
        self.lif1 = snn.Leaky(beta=betas[0], threshold=thresholds[0], init_hidden=True, spike_grad=surrogate.fast_sigmoid(slope=sigmoid_slope))
        self.fc2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.lif2 = snn.Leaky(beta=betas[1], threshold=thresholds[1], init_hidden=True, spike_grad=surrogate.fast_sigmoid(slope=sigmoid_slope))
        self.fc3 = nn.Linear(hidden_layer[1], num_outputs)
        self.lif3 = snn.Leaky(beta=betas[2], threshold=thresholds[2], init_hidden=True, output=True, spike_grad=surrogate.fast_sigmoid(slope=sigmoid_slope))
        self.dp = nn.Dropout(p=p)

        # initializes fully connected layer weights with kaiming uniform distribution
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu') 
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')

        # initializes fully connected layer biases with uniform distribution between -0.1 and 0.1
        nn.init.uniform_(self.fc1.bias, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc2.bias, a=-0.1, b=0.1)
        nn.init.uniform_(self.fc3.bias, a=-0.1, b=0.1)
    
    def forward(self, x, batch_first=False):
        """
        does a forward pass of the model on data x
        
        Inputs:
        - x: data to be passed into model for a forward pass. In the form of (time steps x batch x feature dimension) or
                       (batch x time steps x feature dimension)
        - batch_first: Is True if batch is the first dimension. If not specified, assumes batch_first is False

        Returns: 
        - tensor containing the raw output spike data of shape (time x batch x num_outputs)
        - tensor containing the membrane potential of the output neurons of shape (time x batch x num_outputs)
        """
        # transposes x to the form of (time x batch x flattened feature dimension) if not already in that form
        if(batch_first):
            x = x.transpose(0, 1)
        # x = torch.flatten(x, start_dim=2)

        utils.reset(self)

        # record final layer
        spk_rec = []
        mem_rec = []

        # through the time steps of the data
        for step in range(x.size(0)): # number of time steps in x
            cur1 = self.fc1(x[step])
            spk1 = self.dp(self.lif1(cur1))

            cur2 = self.fc2(spk1)
            spk2 = self.dp(self.lif2(cur2))

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3) 

            spk_rec.append(spk3)
            mem_rec.append(mem3)
        
        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)