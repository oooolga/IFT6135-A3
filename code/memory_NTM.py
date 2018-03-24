from util import *

class Memory(nn.module):
	def __init__(self, N, M):

		super(NTMMemory, self).__init__()

		self.N = N
		self.M = M

		memory = Variable(torch.Tensor(N, M))

		pdb.set_trace()