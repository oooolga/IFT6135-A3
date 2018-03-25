import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from .controller import LSTMController, MLPController
from .memory import Memory
from .heads import Reader, Writer

import ipdb

class NTMCell(nn.Module):
    def __init__(
            self, inp_size, M, N, out_size, controller_size=100, type='lstm'
    ):
        """
        :param inp_size: input dimension
        :param M: memory dimension
        :param N: number of memory
        :param controller_size: controller or hidden size
        :param out_size: output size
        :param type: "lstm" or "mlp"
        """
        super().__init__()
        self.type = type
        self.inp_size = inp_size
        self.M = M
        self.N = N
        self.controller_size = controller_size
        self.out_size = out_size

        self.memory = Memory(N,M)
        self.reader = Reader(controller_size, self.memory)
        self.writer = Writer(controller_size, self.memory)
        if type == "lstm":
            self.controller = LSTMController(inp_size, M, controller_size)
        elif type == "mlp":
            self.controller = MLPController(inp_size, M, controller_size)
        else:
            raise NotImplementedError

        # a learned bias value for previous read initialization
        self.read_init = nn.Parameter(torch.zeros(1, self.memory.M))

        # output decoder
        self.out_dec = nn.Linear(M + controller_size, out_size)

    def reset(self, batch_size):
        """
        reset inner states for a new batch
        """
        self.controller.reset(batch_size)
        self.memory.reset(batch_size)
        self.reader.reset(batch_size)
        self.writer.reset(batch_size)

        self.prev_read = self.read_init.clone().repeat(batch_size, 1)
        self.batch_size = batch_size

    def forward(self, x_t=None):
        """
        :param x_t: [batch_size, inp_dim]
        :return: out: [batch_size, out_size]
        """
        if x_t is None:
            batch_size = self.prev_read.size(0)
            inp_size = self.controller.inp_size
            x_t = Variable(torch.zeros(batch_size, inp_size))
            if torch.cuda.is_available():
                x_t = x_t.cuda()

        o_t = self.controller(x_t, self.prev_read)

        #TODO: Should we perform write first or read first

        self.writer(o_t)
        r_t = self.reader(o_t)
        self.prev_read = r_t

        out = self.out_dec(torch.cat([r_t, o_t], dim=1))
        return F.sigmoid(out)

if __name__ == '__main__':
    inp_size = 9
    M = 28
    N = 100
    controller_size = 100
    out_size = 8
    batch_size = 3


    ntm_cell = NTMCell(inp_size, M, N, out_size, type='mlp')

    ntm_cell.reset(batch_size)
    batch_data = Variable(torch.randn(batch_size, inp_size))
    outs = []
    for _ in range(5):
        outs.append(ntm_cell(batch_data))

    ipdb.set_trace()