import random
from random import randint
from data_gen import CopyTaskGenerator
import ipdb
import torch
from LSTMBaseline import LSTMBaselineCell
from ntm import NTMCell
from torch import optim
import argparse
import utils
import numpy as np

parser = argparse.ArgumentParser("NTM Copy Task")
parser.add_argument("--model", default="baseline",
                    help="[baseline] | [lstm_ntm] | [mlp_ntm] ")
parser.add_argument("--batch-size", default=2)
parser.add_argument("--train-steps", default=1000,
                    help="number of steps to train")
parser.add_argument("--test-interval", default=20)
parser.add_argument("--lr", default=1e-4)
parser.add_argument("--M", default=20)
parser.add_argument("--N", default=128)
parser.add_argument("--controller-size", default=100)

parser.add_argument("--seq-dim", default=8, help="copy task input dim")
parser.add_argument("--max-seq-len", default=20, help="copy task input length")
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

# Copy Task
copy_task_gen = CopyTaskGenerator(
    max_seq_len=args.max_seq_len, seq_dim=args.seq_dim
)

# Load Model
if args.model == 'baseline':
    cell = LSTMBaselineCell(
        input_size=args.seq_dim+1, hidden_size=args.controller_size,
        out_size=args.seq_dim
    )
elif args.model == 'lstm_ntm':
    cell = NTMCell(
        inp_size=args.seq_dim+1,
        out_size=args.seq_dim, M=args.M,
        N=args.N, type='lstm',
        controller_size=args.controller_size
    )
elif args.model == 'mlp_ntm':
    cell = NTMCell(
        inp_size=args.seq_dim+1,
        out_size=args.seq_dim, M=args.M,
        N=args.N, type='lstm',
        controller_size=args.controller_size
    )
else:
    raise NotImplementedError

model = utils.CellWrapper(cell)
if use_cuda:
    model = model.cuda()
print("{} has {} parameters".format(
    args.model, sum([np.prod(p.size()) for p in model.parameters() ])
))

optimizer = optim.RMSprop(
    model.parameters(), momentum=0.9,
    alpha=0.95, lr=args.lr
)

global_step = 0
while global_step < args.train_steps:
    inp, target = copy_task_gen.generate_batch(batch_size=args.batch_size)

    ipdb.set_trace()
    pred = model(inp)



#for i in range(200):
#    optimizer.zero_grad()
#    y_out = baseline_model(inp)
#
#    loss = baseline_model.loss(y_out, target)
#    loss.backward()
#    optimizer.step()
#
#    if i % 20:
#        print(float(loss.data))
