# imports
import snntorch as snn

import torch
import torch.nn as nn

def createSNN(dim_inputs, hidden_layer, num_outputs=2, beta=0.9):
    """
    Function wrapper that initiates a fully connected 3 layer SNN model
    
    Inputs:
    - dim_inputs: dimension size of the input features (flattened)
    - hidden_layer: number of neurons in the hidden layer of the model
    - num_outputs: number of output neurons/classes in the model. Default = 2
    - beta: decay constant for the LIF neurons. Default = 0.9

    Returns: 
    - A fully connected 3 layer spiking neural network with the specified parameters
    """
    return fcSNN(dim_inputs=dim_inputs, hidden_layer=hidden_layer, num_outputs=num_outputs, beta=beta)


class fcSNN(nn.Module):
    def __init__(self, dim_inputs, hidden_layer, num_outputs, beta):
        super().__init__()

        # initializes lif and linear layers for the SNN
        # uses one beta for both lif neuron layers, can be updated if we need more control
        self.fc1 = nn.Linear(dim_inputs, hidden_layer)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(hidden_layer, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
    
    def forward(self, flattened_x, batch_first=False):
        """
        does a forward pass of the model on data x
        
        Inputs:
        - flattened_x: data to be passed into model for a forward pass with the feature dimensions flattened.
                       In the form of (time steps x batch x flattened feature dimension) or
                       (batch x time steps x flattened feature dimension)
        - batch_first: Is True if batch is the first dimension. If not specified, assumes batch_first is False

        Returns: 
        - tensor containing the raw output spike data of shape (time x batch x num_outputs)
        - tensor containing the membrane potential of the output neurons of shape (time x batch x num_outputs)
        """
        x = flattened_x

        # transposes x to the form of (time x batch x flattened feature dimension) if not already in that form
        if(batch_first):
            x = x.transpose(0, 1)

        #initializing hidden states of lif neurons
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # record final layer
        spk2_rec = []
        mem2_rec = []

        # through the time steps of the data
        for step in range(x.size(0)): # number of time steps in x
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)