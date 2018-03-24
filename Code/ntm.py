import torch
from torch import nn

class NTMCell(nn.Module):
    def __init__(self, controller, memory):
        super().__init__()

        self.ctrl = controller
        self.mem = memory

    def reset(self, batch_size):
        self.ctrl.reset(batch_size)
        self.memory.reset(batch_size)
