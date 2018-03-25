import torch
import torch.nn as nn
import ipdb

class CellWrapper(nn.Module):
    """
    A wrapper for single cell to handle input output for copy task
    """
    def __init__(self, cell):
        super().__init__()
        self.cell = cell

    def forward(self, inp):
        """
        :param inp: [T+1, batch_size, inp_size]
        :return: out [T, batch_size, out_size]
        """
        # reset cell and clear memory
        cell = self.cell
        cell.reset(inp.size(1))

        # read in all input first
        # no need to record output. Just update memory
        for t in range(inp.size(0)):
            cell.forward(inp[t])

        # start outputting (no input)
        # read from memory and start copying
        out = []
        for t in range(inp.size(0)-1):
            out.append(cell.forward(None))
        out = torch.stack(out)
        return out