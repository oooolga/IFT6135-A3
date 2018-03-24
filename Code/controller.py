import torch
from torch import nn

class Controller(nn.Module):
    def __init__(self, inp_size, controller_size=100):
        super().__init__()
        self.inp_size = inp_size
        self.controller_size = controller_size

    def reset(self, batch_size):
        """
        Set the internal state of a controller. If there is one
        """
        raise NotImplementedError

class MLPController(Controller):
    """
    A MLP controller
    """
    def __init__(self, inp_size, controller_size=100):
        super().__init__(inp_size, controller_size)

        # A Simple MLP with 100 hidden units
        self.fc1 = nn.Linear(inp_size, controller_size)

        # init fc1 here possibily


    def forward(self, inp):
        """
        :param inp: [batch_size, inp_size]
        :return: out: [batch_size, controller_size]
        """
        return self.fc1(inp)

    def reset(self, batch_size):
        """
        Nothing to do here
        """
        return

class LSTMController(Controller):
    """
    An LSTM controller
    """
    def __init__(self, inp_size, controller_size=100):
        super().__init__(inp_size, controller_size)

        self.lstm = nn.LSTMCell(input_size=inp_size, hidden_size=controller_size)
        self.h_init = nn.Parameter(torch.zeros(1, 1, controller_size))
        self.c_init = nn.Parameter(torch.zeros(1, 1, controller_size))

    def forward(self, inp, prev_state):
        """
        :param inp: [batch_size, inp_size]
        :return: out: [batch_size, controller_size], state
        """
        state = self.lstm(inp, prev_state)
        return state[0], state

    def reset(self, batch_size):
        state = (
            self.h_init.clone().repeat(1, batch_size, 1),
            self.c_init.clone().repeat(1, batch_size, 1)
        )
        return state

