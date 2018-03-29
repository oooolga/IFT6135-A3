import random
from random import randint
from data_gen import CopyTaskGenerator
import ipdb, pdb
import torch
from models import LSTMBaselineCell
from models import NTMCell
from torch import optim
import argparse, os
import utils
import numpy as np
import os

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser("Copy Task Test")
parser.add_argument('--model-name', type=str,
                    help='name of the model to load')
args = parser.parse_args()

model, optimizer, args = utils.load_checkpoint(args.model_name, use_cuda)

# Copy Task
copy_task_gen = CopyTaskGenerator(
    max_seq_len=args.max_seq_len, seq_dim=args.seq_dim
)

criterion = torch.nn.BCELoss()

################
'''
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
pdb.set_trace()
'''


