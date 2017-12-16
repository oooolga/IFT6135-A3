import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

import pdb
OFFSET = 1e-16

class Memory(nn.Module):
    def __init__(self, N, M):

        super(Memory, self).__init__()

        self.N = N
        self.M = M

        # a learned initialized memory value N x M

        self.mem_init = nn.Parameter(torch.zeros(N,M))
        nn.init.xavier_uniform(self.mem_init, 1)

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.memory = self.mem_init.clone().repeat(batch_size, 1, 1)

    def reading(self, w_t):
        # w_t = Batch x N
        r_t = torch.matmul(w_t.unsqueeze(1), self.memory).squeeze(1)
        return r_t

    def writing(self, w_t, e_t, a_t):

        M_hat_t = self.memory * (1 - torch.matmul(w_t.unsqueeze(2),
                                                  e_t.unsqueeze(1)))
        self.memory = M_hat_t + torch.matmul(w_t.unsqueeze(2),
                                             a_t.unsqueeze(1))

    def content_addressing(self, k_t, beta_t, g_t, s_t, gamma_t, w_t_minus_1):
        """
        :param k_t: [batch_size, M]
        :param beta_t: [batch_size, 1]
        :param g_t: [batch_size, 1]
        :param s_t: [batch_size, 3]
        :param gamma_t: a number
        :param w_t_minus_1: [batch_size, N]
        :return:
        """
        #import ipdb
        #ipdb.set_trace()

        k_t = k_t.view(self.batch_size, 1, self.M)
        beta_t = beta_t.view(self.batch_size, 1).repeat(1,self.N)
        K = F.cosine_similarity(self.memory+ OFFSET, k_t+ OFFSET, dim=-1)
        w_c_t = F.softmax(beta_t * K, dim=1)

        # g_t = Batch x 1
        # w_c_t = Batch x N
        g_t = g_t.view(self.batch_size, 1).repeat(1,self.N)
        w_g_t = g_t*w_c_t + (1-g_t)*w_t_minus_1

        def _circular_convolution(w_g_t, s_t):
            # I didn't have torch.Tensor
            w_tilde_t = Variable(torch.zeros(self.batch_size, self.N))
            if torch.cuda.is_available():
                w_tilde_t = w_tilde_t.cuda()

            kern_size = s_t.size(1)
            # circular padding
            w_g_t_padded = w_g_t.clone()
            w_g_t_padded = w_g_t_padded.repeat(1,3)
            w_g_t_padded = w_g_t_padded[:, self.N-(kern_size//2):-(self.N-(kern_size//2))]
            if kern_size % 2 == 0:
                w_g_t_padded = w_g_t_padded[:, :-1]

            w_g_t_padded = w_g_t_padded.unsqueeze(1)

            #w_tilde_t = F.conv1d(w_g_t_padded, s)
            for batch_idx in range(self.batch_size):
                w_tilde_t[batch_idx] = F.conv1d(
                    w_g_t_padded[batch_idx].view(1,1,-1),
                    s_t[batch_idx].view(1,1,-1)
                ).view(-1)
            return w_tilde_t

        w_tilde_t = _circular_convolution(w_g_t, s_t)
        gamma_t = gamma_t.view(self.batch_size, 1).repeat(1,self.N)
        w_t = w_tilde_t ** gamma_t
        w_t_sum = torch.sum(w_t, 1)
        w_t_sum = w_t_sum.view(self.batch_size, 1).repeat(1,self.N) + OFFSET
        return w_t / w_t_sum


if __name__ == '__main__':
    N = 20
    M = 15
    batch_size = 3

    temp = Memory(N, M)
    temp.reset(batch_size)

    w_t = Variable(torch.Tensor(batch_size, N))
    e_t = Variable(torch.Tensor(batch_size, M))
    a_t = Variable(torch.Tensor(batch_size, M))
    k_t = Variable(torch.Tensor(batch_size, M))
    beta_t = Variable(torch.Tensor(batch_size))
    g_t = Variable(torch.Tensor(batch_size))
    s_t = Variable(torch.Tensor(batch_size, 3))
    gamma_t = Variable(torch.Tensor(batch_size))

    temp.reading(w_t)
    temp.writing(w_t, e_t, a_t)
    temp.content_addressing(k_t, beta_t, g_t, s_t, gamma_t, w_t)
    print("pass memory test")
