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
parser.add_argument('--lstm-model', type=str,
                    help='name of the lstm model to load')
parser.add_argument('--lstm-ntm-model', type=str,
                    help='name of the lstm-ntm model to load')
parser.add_argument('--mlp-ntm-model', type=str,
                    help='name of the mlp-ntm model to load')
parser.add_argument('--plot-dir', type=str, default='./plots',
                    help='directory for plots')
parser.add_argument('--plot-name', type=str, default='avgloss_vs_t')
org_args = parser.parse_args()

lstm_model, _, lstm_args, _ = utils.load_checkpoint(org_args.lstm_model, use_cuda)
lstm_ntm_model, _, lstmn_ntm_args, _ = utils.load_checkpoint(org_args.lstm_ntm_model, use_cuda)
mlp_ntm_model, _, mplp_ntm_args, _ = utils.load_checkpoint(org_args.mlp_ntm_model, use_cuda)

models = {'lstm': {'model':lstm_model, 'losses':[], 'color':'ro'},
		  'lstm-ntm': {'model':lstm_ntm_model, 'losses':[], 'color':'bs'},
		  'mlp-ntm':{'model':mlp_ntm_model, 'losses':[], 'color':'g^'}}


if not os.path.exists(org_args.plot_dir):
    os.makedirs(org_args.plot_dir)

# Copy Task
copy_task_gen = CopyTaskGenerator(
    max_seq_len=100, seq_dim=lstm_args.seq_dim
)

criterion = torch.nn.BCELoss()
losses = {}

for model_name in models.keys():
	models[model_name]['model'].eval()

for T in range(10, 101, 10):
	inp, target = copy_task_gen.generate_batch(20, T)
	for model_name in models.keys():
		pred = models[model_name]['model'](inp)
		loss = criterion(pred, target)
		models[model_name]['losses'].append(loss.cpu().data[0])

for model_name in models.keys():
	plt.plot(list(range(10, 101, 10)), models[model_name]['losses'], models[model_name]['color'],
			 label=model_name)
plt.xlabel('T')
plt.ylabel('avg loss')

plt.title('average loss vs. T')
plt.savefig(os.path.join(org_args.plot_dir, org_args.plot_name+'.png'))
plt.clf()

