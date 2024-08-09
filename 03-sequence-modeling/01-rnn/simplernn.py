import torch 
from torch import nn
from torch functiona L # ADD THE JAX 



class RNNModel(nn.Module):


    def __init__(self, rnn_layer, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        if self.rnn.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.linear = nn.Linear(
            self.num_directions * self.rnn.hidden_size, self.rnn.input_size)

    # forward

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.rnn.input_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, begin_state


    #begin state

    def begin_state(self, device, batch_size=1):
        tensor = torch.zeros((self.num_directions * self.rnn.num_layers, 
                              batch_size, self.rnn.hidden_size), 
                             device=device)
        if isinstance(self.rnn, nn.LSTM):
            return (tensor, tensor) 
        else:
            return tensor
