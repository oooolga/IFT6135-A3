import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from util import *

import pdb

class NTMMemory(nn.Module):
	def __init__(self, N, M, batch_size):

		super(NTMMemory, self).__init__()

		self.N = N
		self.M = M
		self.batch_size = batch_size

		memory = Variable(torch.Tensor(N, M))
		torch.nn.init.constant(memory, 0)

		# M = Batch x N x M
		self.memory = memory.repeat(batch_size, 1, 1)


	def reading(self, w_t):
		# w_t = Batch x N
		r_t = torch.matmul(w_t.unsqueeze(1), self.memory).squeeze(1)
		return r_t

	def writing(self, w_t, e_t, a_t):
		
		#prev_memory = self.memory.clone()
		M_hat_t = self.memory * (1 - torch.matmul(w_t.unsqueeze(2),
												  e_t.unsqueeze(1)))
		self.memory = M_hat_t + torch.matmul(w_t.unsqueeze(2),
											 a_t.unsqueeze(1))
		
	def content_addressing(self, k_t, beta_t, g_t, w_t_minus_1, s_t, gamma_t):
		# beta_t = Batch x 1
		# k_t = Batch x M
		k_t = k_t.view(self.batch_size, 1, self.M)
		beta_t = beta_t.view(self.batch_size, 1).repeat(1,self.N)
		K = F.cosine_similarity(self.memory+ OFFSET, k_t+ OFFSET, dim=-1)
		w_c_t = F.softmax(beta_t * K, dim=1)

		# g_t = Batch x 1
		# w_c_t = Batch x N
		g_t = g_t.view(self.batch_size, 1).repeat(1,self.N)
		w_g_t = g_t*w_c_t + (1-g_t)*w_t_minus_1

		# TODO: to do shift
		w_tilde_t = w_t_minus_1

		gamma_t = gamma_t.view(self.batch_size, 1).repeat(1,self.N)
		w_t = w_tilde_t ** gamma_t
		w_t_sum = torch.sum(w_t, 1)
		w_t_sum = w_t_sum.view(self.batch_size, 1).repeat(1,self.N) + OFFSET
		return w_t_sum