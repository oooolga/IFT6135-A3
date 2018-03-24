#####################################
# This is the data generating script
#####################################
import torch
import ipdb
from random import randint

class CopyTaskGenerator(object):
    def __init__(self, max_seq_len=20, seq_dim=8):
        self.max_seq_len = max_seq_len
        self.seq_dim = seq_dim

    def generate(self, batch_size, T=None):
        """
        generate a batch of sequences with input and output.
        :param batch_size: batch_size
        :param T: length of each sequence. Should be less than max_seq_len. If None then randomly sample
        from [1, T].
        :return: (input, output).
        input: [T+1, batch_size, seq_dim+1].
        output: [T, batch_size, seq_dim]
        """
        if T is None:
            T = randint(1, self.max_seq_len)
        assert(T <= self.max_seq_len, "Only allowed {} lenghs sampled".format(self.max_seq_len))

        seqs = torch.rand(T, batch_size, self.seq_dim)
        input = torch.cat(
            [
                torch.cat([seqs, torch.zeros(T, batch_size, 1)], 2),
                torch.cat([torch.zeros(1, batch_size, self.seq_dim), torch.ones(1, batch_size, 1)], 2)
            ]
        )
        output = seqs.clone()
        return input, output

# Testing
import ipdb
ipdb.set_trace()
data_gen = CopyTaskGenerator()
inp, out = data_gen.generate(1, 5)