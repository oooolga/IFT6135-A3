from torch.autograd import Variable
import torch
import ipdb
from ntm_cell import NTMCell

##### CONFIG #######
inp_size = 9
M = 28
N = 100
controller_size = 100
out_size = 8

batch_size = 3
####################

ntm_cell = NTMCell(inp_size, M, N, out_size, type='mlp')

ntm_cell.reset(batch_size)
batch_data = Variable(torch.randn(batch_size, inp_size))
outs = []
for _ in range(5):
    outs.append(ntm_cell(batch_data))

ipdb.set_trace()
