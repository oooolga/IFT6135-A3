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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser("Copy Task Test")
parser.add_argument('--model-name', type=str,
                    help='name of the model to load')
parser.add_argument('--plot-dir', type=str, default='./plots',
                    help='directory for plots')
parser.add_argument('--plot-name', type=str, default='avgloss_vs_t')
org_args = parser.parse_args()

model, _, args, _ = utils.load_checkpoint(org_args.model_name, use_cuda)

if not os.path.exists(org_args.plot_dir):
    os.makedirs(org_args.plot_dir)

# Copy Task
copy_task_gen = CopyTaskGenerator(
    max_seq_len=100, seq_dim=args.seq_dim
)

losses = []

criterion = torch.nn.BCELoss()
model.eval()

for T in range(10, 101, 10):
	inp, target = copy_task_gen.generate_batch(20, T)
	pred = model(inp)
	loss = criterion(pred, target)
	losses.append(loss.cpu().data[0])

plt.plot(list(range(10, 101, 10)), losses, 'ro')
plt.xlabel('T')
plt.ylabel('avg loss')

plt.title('average loss vs. T')
plt.savefig(os.path.join(org_args.plot_dir, org_args.plot_name+'.png'))
plt.clf()




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


