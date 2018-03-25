import torch
from torch import nn

class Controller(nn.Module):
    def __init__(self, inp_size, M, controller_size=100):
        super().__init__()
        self.inp_size = inp_size
        self.M = M
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
    def __init__(self, inp_size, M, controller_size=100):
        super().__init__(inp_size, M, controller_size)

        # A Simple MLP with 100 hidden units
        self.fc1 = nn.Linear(inp_size+M, controller_size)

        # init fc1 here possibily


    def forward(self, inp, prev_read):
        """
        :param inp: [batch_size, inp_size]
        :param prev_read: [batch_size, M]
        :return: out: [batch_size, controller_size]
        """
        return self.fc1(torch.cat([inp, prev_read]))

    def reset(self, batch_size):
        """
        Nothing to do here
        """
        return

class LSTMController(Controller):
    """
    An LSTM controller
    """
    def __init__(self, inp_size, M, controller_size=100):
        super().__init__(inp_size, M, controller_size)

        self.lstm = nn.LSTMCell(input_size=inp_size+M, hidden_size=controller_size)
        self.h_init = nn.Parameter(torch.zeros(1, controller_size))
        self.c_init = nn.Parameter(torch.zeros(1, controller_size))

    def reset(self, batch_size):
        """
        reset current_state to something new
        """
        self.state = (
            self.h_init.clone().repeat(batch_size, 1),
            self.c_init.clone().repeat(batch_size, 1)
        )

    def forward(self, inp, prev_read):
        """
        :param inp: [batch_size, inp_size]
        :param prev_read: [batch_size, M]
        :return: out: [batch_size, controller_size]
        """
        self.state = self.lstm(torch.cat([inp, prev_read], 1), self.state)
        return self.state[0]


