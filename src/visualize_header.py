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
parser.add_argument('--model-name', type=str, default='./saved_models/mlpntm_17500.pt',
                    help='name of the model to load')
parser.add_argument('--plot-dir', type=str, default='./plots',
                    help='directory for plots')
parser.add_argument('--in-out-name', type=str, default='mlpntm_inout')
parser.add_argument('--head-name', type=str, default='mlpntm_head')
parser.add_argument('--attn-name', type=str, default='mlpntm_attn')
parser.add_argument('--plot-name', type=str, default='mlpntm_visualization')
org_args = parser.parse_args()

model, _, args, _ = utils.load_checkpoint(org_args.model_name, use_cuda)
model.set_weight_plot_flag(True)
model2, _, _, _ = utils.load_checkpoint(org_args.model_name, use_cuda)
model2.set_weight_plot_flag(True)

if not os.path.exists(org_args.plot_dir):
    os.makedirs(org_args.plot_dir)

# Copy Task
copy_task_gen = CopyTaskGenerator(
    max_seq_len=10, seq_dim=args.seq_dim
)

model.eval()
T = 10
inp, target = copy_task_gen.generate_batch(1, T)
_ = model(inp)

inp = inp.cpu().squeeze(1).data.numpy().swapaxes(0,1)
inp = np.concatenate((inp,np.zeros((9, T))), 1)
target = target.cpu().squeeze(1).data.numpy().swapaxes(0,1)
target = np.concatenate((np.zeros((8,T+1)), target), 1)
target = np.concatenate((target, np.zeros((1,2*T+1))), 0)

T = 100
inp2, target2 = copy_task_gen.generate_batch(1, T)
_ = model2(inp2)

inp2 = inp2.cpu().squeeze(1).data.numpy().swapaxes(0,1)
inp2 = np.concatenate((inp2,np.zeros((9, T))), 1)
target2 = target2.cpu().squeeze(1).data.numpy().swapaxes(0,1)
target2 = np.concatenate((np.zeros((8,T+1)), target2), 1)
target2 = np.concatenate((target2, np.zeros((1,2*T+1))), 0)

utils.plot_visualize_head(model, inp, target,
						  os.path.join(org_args.plot_dir, org_args.in_out_name+'_10.png'),
						  os.path.join(org_args.plot_dir, org_args.head_name+'_10.png'),
						  os.path.join(org_args.plot_dir, org_args.attn_name+'_10.png'))

utils.plot_visualize_head(model2, inp2, target2,
						  os.path.join(org_args.plot_dir, org_args.in_out_name+'_100.png'),
						  os.path.join(org_args.plot_dir, org_args.head_name+'_100.png'),
						  os.path.join(org_args.plot_dir, org_args.attn_name+'_100.png'))

utils.plot_visualize_2(model, inp, target, model2, inp2, target2,
					  os.path.join(org_args.plot_dir, org_args.plot_name+'.png'))