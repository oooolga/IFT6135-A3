import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def _split_cols(input, lengths):
    """
    :param input: [ batch_size, dim ]
    :param lengths: a list of lengths
    :return: split the columns
    """
    results = []
    start_idx = 0
    for length in lengths:
        results.append(input[:, start_idx:start_idx+length])
        start_idx += length
    return tuple(results)

def _softplus(x):
    return torch.log( 1 + torch.exp(x) )

def _sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class Reader(nn.Module):
    def __init__(self, controller_size, memory):
        super().__init__()
        # NTMMemory object
        self.memory = memory

        # decode addressing argument
        self.fc = nn.Linear(controller_size, memory.M + 1 + 1 + 3 + 1)
        self.address_param_lengths = [memory.M, 1, 1, 3, 1]

    def reset(self, batch_size):
        return Variable(torch.zeros(batch_size, self.memory.N))


    def forward(self, controller_out, prev_w):
        """
        :param controller_out: [bsz, controller_size]
        :param prev_w: [bsz, N]
        :return: r_t [bsz, M], w_t [bsz, N]
        """
        k, beta, g, s, gamma = _split_cols(self.address_param_lengths)
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=-1)
        gamma = F.relu(gamma) + 1

        w_t = self.memory.content_address()

        return self.memory.reading(w_t), w_t