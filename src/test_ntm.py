from models import NTMCell
import torch
from torch.autograd import Variable
import ipdb

inp_size = 9
controller_size = 100
out_size = 8
batch_size = 1
M = 20
N = 128

ntm_cell = NTMCell(inp_size, M, N, out_size, type='mlp')

ntm_cell.reset(batch_size)
batch_data = Variable(torch.randn(batch_size, inp_size))
outs = []
for _ in range(20):
    outs.append(ntm_cell(batch_data))
ipdb.set_trace()


