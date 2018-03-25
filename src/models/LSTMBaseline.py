import torch
import torch.nn as nn
import pdb
from torch.autograd import Variable

class LSTMBaselineCell(nn.Module):

    def __init__(self, input_size, hidden_size, out_size):
        super(LSTMBaselineCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        # learned initialize value for LSTM
        self.h_init = nn.Parameter(torch.zeros(1, hidden_size))
        self.c_init = nn.Parameter(torch.zeros(1, hidden_size))

        self.lstm = nn.LSTMCell(
            self.input_size, self.hidden_size
        )
        self.out_dec = nn.Linear(self.hidden_size, self.out_size)
        self.sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()


    def reset(self, batch_size):
        self.state = (
            self.h_init.clone().repeat(batch_size, 1),
            self.c_init.clone().repeat(batch_size, 1)
        )


    def forward(self, input=None):
        """
        Single step forward
        :param input: [batch_size, inp_size]
        :return: out: [batch_size, out_size]
        """
        if input is None:
            batch_size = self.state[0].size(0)
            inp_size = self.input_size
            input = Variable(torch.zeros(batch_size, inp_size))
            if torch.cuda.is_available():
                input = input.cuda()

        self.state = self.lstm(input, self.state)
        out = self.sigmoid(
            self.out_dec(self.state[0])
        )
        return out
