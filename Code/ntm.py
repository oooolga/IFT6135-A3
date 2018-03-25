import torch
from torch import nn
from util import use_cuda
from torch.autograd import Variable

class NTMCell(nn.Module):
    def __init__(self, controller, memory, reader, writer):
        """
        :param controller: Controller object
        :param memory: NTMMomory object
        :param reader: Reader object
        :param writer: Writer object
        """
        super().__init__()
        self.controller = controller
        self.memory = memory
        self.reader = reader
        self.writer = writer

        # a learned bias value for previous read initialization
        self.read_init = nn.Parameter(torch.zeros(1, self.mem.M))

    def init_sequences(self, batch_size):
        """
        reset inner states for a new batch
        """
        self.ctrl.reset(batch_size)
        self.memory.reset(batch_size)
        self.reader.reset(batch_size)
        self.writer.reset(batch_size)

        self.prev_read = self.read_init.clone().repeat(batch_size, 1)
        self.batch_size = batch_size

    def forward(self, x_t=None):
        """
        :param x_t: [batch_size, inp_dim]
        :return:
        """
        if x_t is None:
            batch_size = self.prev_read.size(0)
            inp_size = self.controller.inp_size
            x_t = Variable(torch.zeros(batch_size, inp_size))
            if use_cuda:
                x_t = x_t.cuda()

        o_t = self.controller(x_t, self.prev_read)


        #TODO: Should we perform write first or read first
        r_t = self.reader(o_t)

