#####################################
# This is the data generating script
#####################################
import torch
from torch.autograd import Variable
import pdb
from random import randint
import numpy as np

class CopyTaskGenerator(object):
    def __init__(self, max_seq_len=20, seq_dim=8):
        self.max_seq_len = max_seq_len
        self.seq_dim = seq_dim

    def generate_batch(self, batch_size, T=None):
        """
        generate a batch of sequences with input and output.
        :param batch_size: batch_size
        :param T: length of each sequence. If None then randomly sample from [1, T].
        :return: (input, output).
        input: [T+1, batch_size, seq_dim+1].
        output: [T, batch_size, seq_dim]
        """
        if T is None:
            T = randint(1, self.max_seq_len)

        #seqs = torch.rand(T, batch_size, self.seq_dim)
        seqs = np.random.randint(2, size=(T, batch_size, self.seq_dim))
        seqs = Variable(torch.from_numpy(seqs))
        input = torch.cat(
            [
                torch.cat([seqs, torch.zeros(T, batch_size, 1).type_as(seqs)], 2),
                torch.cat([torch.zeros(1, batch_size, self.seq_dim).type_as(seqs),
                           torch.ones(1, batch_size, 1).type_as(seqs)], 2)
            ]
        )
        output = seqs.clone()
        if torch.cuda.is_available():
            return input.float().cuda(), output.float().cuda()
        else:
            return input.float(), output.float()

