import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import pdb

class NTMMemory(nn.Module):

    def __init__(self, N, M, batch_size):

        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M
        self.batch_size = batch_size

        # a learned memory initialize value
        self.mem_init = nn.Parameter(torch.Tensor(N, M))
        torch.nn.init.constant(self.mem_init, 0)

        # [ Batch, N, M ]
        self.memory = None

    def reset(self, batch_size):
        self.memory = self.mem_init.repeat(batch_size, 1, 1)


    def reading(self, w_t):
        """
        :param w_t: [batch_size, N]
        :return: r_t: [batch_size, M]
        """
        assert(self.memory is not None, "reinitialize memory first!")
        r_t = torch.matmul(w_t.unsqueeze(1), self.memory).squeeze(1)
        return r_t

    def writing(self, w_t, e_t, a_t):
        """
        Modify the self.memory
        :param w_t: [batch_size, N]
        :param e_t: [batch_size, N]
        :param a_t: [batch_size, N]
        """
        assert(self.memory is not None, "reinitialize memory first!")
        M_hat_t = self.memory * (1 - torch.matmul(w_t.unsqueeze(2),
                                                  e_t.unsqueeze(1)))
        self.memory = M_hat_t + torch.matmul(w_t.unsqueeze(2),
                                             a_t.unsqueeze(1))

    def addressing(self, prev_w, controller_out):
        """
        Use the four stage addressing mechanism to generate the next attention weight.
        :param prev_w: previous weight. (could be from reader or writer) [batch_size, N]
        
        :return:
        """


        # beta_t = Batch x 1
        # k_t = Batch x M
        k_t = k_t.view(self.batch_size, 1, self.M)
        w_c_t = F.softmax(beta_t * \
                          F.cosine_similarity(self.memory, k_t, dim=-1), dim=1)
