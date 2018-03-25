from ntm.controller import LSTMController
from ntm.heads import Reader, Writer
from ntm.memoryNTM import NTMMemory
from ntm import NTMCell
from torch.autograd import Variable
import torch
import ipdb

##### CONFIG #######
inp_size = 9
M = 28
N = 100
controller_size = 100
out_size = 8

batch_size = 3
####################

memory = NTMMemory(N, M)
reader = Reader(controller_size, memory)
writer = Writer(controller_size, memory)
controller = LSTMController(inp_size, M, controller_size)
ntm_cell = NTMCell(controller, memory, reader, writer, out_size)


ntm_cell.reset(batch_size)
batch_data = Variable(torch.randn(batch_size, inp_size))
outs = []
for _ in range(5):
    outs.append(ntm_cell(batch_data))

ipdb.set_trace()
