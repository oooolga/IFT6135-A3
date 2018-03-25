from util import *
from memoryNTM import *

if __name__ == '__main__':
	N = 20
	M = 15
	batch_size = 30

	temp = NTMMemory(N, M, batch_size)

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
	temp.content_addressing(k_t, beta_t, g_t, w_t, s_t, gamma_t)