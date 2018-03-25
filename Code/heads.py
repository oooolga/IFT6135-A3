import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from util import use_cuda

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


class Head(nn.Module):
    def __init__(self, controller_size, memory):
        super(Head, self).__init__()
        self.controller_size = controller_size
        # NTMMemory object
        self.memory = memory

        # decode addressing argument
        self.address_param_decoder = nn.Linear(controller_size, memory.M + 1 + 1 + 3 + 1)
        nn.init.xavier_uniform(self.address_param_decoder.weight, gain=1.0)
        self.address_param_decoder.bias.data.zero_()
        self.address_param_lengths = [memory.M, 1, 1, 3, 1]

    def reset(self, batch_size):
        if use_cuda:
            self.prev_w = Variable(torch.ones(batch_size, self.memory.N) / self.memory.N).cuda()
        else:
            self.prev_w = Variable(torch.ones(batch_size, self.memory.N) / self.memory.N)


    def _get_w_t(self, controller_out):
        """
        :param controller_out: [bsz, controller_size]
        :param w_t: [bsz, N]
        """
        k, beta, g, s, gamma = _split_cols(
            self.address_param_decoder(controller_out),
            self.address_param_lengths
        )
        k = k.contiguous()
        beta = F.softplus(beta)
        g = F.sigmoid(g)
        s = F.softmax(s, dim=-1)
        gamma = F.relu(gamma) + 1

        w_t = self.memory.content_addressing(k, beta, g, s, gamma, self.prev_w)
        return w_t



class Reader(Head):
    def __init__(self, controller_size, memory):
        super().__init__(controller_size, memory)

    def forward(self, controller_out):
        """
        :param controller_out: [bsz, controller_size]
        :return: r_t [bsz, M]
        """
        w_t = self._get_w_t(controller_out)
        # update state
        self.prev_w = w_t
        return self.memory.reading(w_t)


class Writer(Head):
    def __init__(self, controller_size, memory):
        super().__init__(controller_size, memory)
        self.writer_decoder = nn.Linear(self.controller_size, 2 * self.memory.M)
        nn.init.xavier_uniform(self.writer_decoder.weight, gain=1.0)
        self.writer_decoder.bias.data.zero_()

    def forward(self, controller_out):
        """
        change the memory
        :param controller_out: [bsz, controller_size]
        """
        w_t = self._get_w_t(controller_out)
        # update state
        self.prev_w = w_t
        e_t, a_t = _split_cols(
            self.writer_decoder(controller_out),
            [self.memory.M, self.memory.M]
        )
        e_t = F.sigmoid(e_t)
        self.memory.writing(w_t, e_t, a_t)
