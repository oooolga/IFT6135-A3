from data_gen import CopyTaskGenerator
import ipdb, pdb
import torch
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
parser.add_argument('--model-name', type=str, default='./saved_models/lstm_ntm_2500.pt',
                    help='name of the model to load')
parser.add_argument('--plot-dir', type=str, default='./plots',
                    help='directory for plots')
parser.add_argument('--plot-name', type=str, default='head_visualization')
org_args = parser.parse_args()

model, _, args, _ = utils.load_checkpoint(org_args.model_name, use_cuda)
model.set_weight_plot_flag(True)

if not os.path.exists(org_args.plot_dir):
    os.makedirs(org_args.plot_dir)

# Copy Task
copy_task_gen = CopyTaskGenerator(
    max_seq_len=10, seq_dim=args.seq_dim
)

model.eval()
inp, target = copy_task_gen.generate_batch(1, 10)
pred = model(inp)

utils.plot_visualize_head(model, os.path.join(org_args.plot_dir, org_args.plot_name+'.png'))