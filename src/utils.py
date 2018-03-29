import torch
import torch.nn as nn
import ipdb, pdb
import os
import tensorflow as tf
import numpy as np
from models import LSTMBaselineCell
from models import NTMCell
from torch import optim
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec


class CellWrapper(nn.Module):
    """
    A wrapper for single cell to handle input output for copy task
    """
    def __init__(self, cell):
        super().__init__()
        self.cell = cell
        self.R = None
        self.plot_weight_flag = False

    def set_weight_plot_flag(self, flag):
        self.plot_weight_flag = flag

    def forward(self, inp):
        """
        :param inp: [T+1, batch_size, inp_size]
        :return: out [T, batch_size, out_size]
        """
        # reset cell and clear memory
        cell = self.cell
        cell.reset(inp.size(1))

        if self.plot_weight_flag:
            self.write_w = np.empty((0, cell.N))
            self.read_w = np.empty((0, cell.N))
            self.read_head = np.empty((0, cell.M))
            self.write_head = np.empty((0, cell.M))

        # read in all input first
        # no need to record output. Just update memory
        for t in range(inp.size(0)):
            cell.forward(inp[t])
            if self.plot_weight_flag:
                write_w = cell.writer.prev_w.cpu().data.numpy()
                write_head = cell.writer.a_t.cpu().data.numpy()
                self.write_w = np.concatenate((self.write_w, write_w))
                self.write_head = np.concatenate((self.write_head, write_head))

                read_w = cell.reader.prev_w.cpu().data.numpy()
                read_head = cell.r_t.cpu().data.numpy()
                self.read_w = np.concatenate((self.read_w, read_w))
                self.read_head = np.concatenate((self.read_head, read_head))

        # start outputting (no input)
        # read from memory and start copying
        out = []
        for t in range(inp.size(0)-1):
            out.append(cell.forward(None))
            if self.plot_weight_flag:
                write_w = cell.writer.prev_w.cpu().data.numpy()
                write_head = cell.writer.a_t.cpu().data.numpy()
                self.write_w = np.concatenate((self.write_w, write_w))
                self.write_head = np.concatenate((self.write_head, write_head))

                read_w = cell.reader.prev_w.cpu().data.numpy()
                read_head = cell.r_t.cpu().data.numpy()
                self.read_w = np.concatenate((self.read_w, read_w))
                self.read_head = np.concatenate((self.read_head, read_head))

        out = torch.stack(out)

        if self.plot_weight_flag:
            self.write_w = np.swapaxes(self.write_w,0,1)
            self.read_w = np.swapaxes(self.read_w,0,1)
            self.read_head = np.swapaxes(self.read_head,0,1)
            self.write_head = np.swapaxes(self.write_head,0,1)

        return out

class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

def get_model_optimizer(args, use_cuda):
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
            N=args.N, type='mlp',
            controller_size=args.controller_size
        )
    else:
        raise NotImplementedError

    model = CellWrapper(cell)
    if use_cuda:
        model = model.cuda()

    optimizer = optim.RMSprop(
        model.parameters(), momentum=args.momentum,
        lr=args.lr, alpha=args.alpha
    )

    return model, optimizer

def save_checkpoints(state, model_name):
    torch.save(state, model_name)
    print('Finished saving model: {}'.format(model_name))

def load_checkpoint(model_name, use_cuda):
    if model_name and os.path.isfile(model_name):
        checkpoint = torch.load(model_name)
        args = checkpoint['args']
        model, optimizer = get_model_optimizer(args, use_cuda)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Finished loading model and optimizer from {}'.format(model_name))
    else:
        print('File {} not found.'.format(model_name))
        raise FileNotFoundError
    return model, optimizer, args, checkpoint['global_step']

def plot_visualize_head(model, inp, tgt, in_tgt_path, head_path, attn_path):
    #rescale
    write_head, read_head = rescale_heads(model)

    heads = {'write':write_head, 'read':read_head}
    attentions = {'write':model.write_w, 'read':model.read_w} 

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    im = ax1.imshow(inp, vmin=0, vmax=1, interpolation='nearest', cmap='gray')
    ax1.set_xlabel('inputs')

    im = ax2.imshow(tgt, vmin=0, vmax=1, interpolation='nearest', cmap='gray')
    ax2.set_xlabel('outputs')
    plt.title('inputs and outputs')
    plt.savefig(in_tgt_path)
    plt.clf()
    print('Image {} is saved.'.format(in_tgt_path))


    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    im = ax1.imshow(heads['write'], vmin=0, vmax=1, interpolation='nearest')
    ax1.set_ylabel('adds')

    im = ax2.imshow(heads['read'], vmin=0, vmax=1, interpolation='nearest')
    ax2.set_ylabel('reads')
    plt.title('the vectors add to/read from memory')
    plt.savefig(head_path)
    plt.clf()
    print('Image {} is saved.'.format(head_path))

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    im = ax1.imshow(attentions['write'], vmin=0, vmax=1, interpolation='nearest', cmap='gray')
    ax1.set_ylabel('location')
    ax1.set_xlabel('time')

    im = ax2.imshow(attentions['read'], vmin=0, vmax=1, interpolation='nearest', cmap='gray')
    ax2.set_xlabel('time')
    plt.title('read/write weighting')
    plt.savefig(attn_path)
    plt.clf()
    print('Image {} is saved.'.format(attn_path))


def rescale_heads(model):
    #rescale
    write_head, read_head = model.write_head, model.read_head
    max_write, max_read = np.max(write_head), np.max(read_head)
    min_write, min_read = np.min(write_head), np.min(read_head)
    scale_write, scale_read = np.abs(max_write-min_write), np.abs(max_read-min_read)
    write_head = (write_head-min_write)/scale_write
    return write_head, read_head

def plot_visualize_2(model, inp, tgt, model2, inp2, tgt2, out_path):
    write_head, read_head = rescale_heads(model)
    write_head2, read_head2 = rescale_heads(model2)
    heads = {'write':write_head, 'read':read_head}
    attentions = {'write':model.write_w, 'read':model.read_w}
    heads2 = {'write':write_head2, 'read':read_head2}
    attentions2 = {'write':model2.write_w, 'read':model2.read_w} 

    in_h, head_h, attn_h = inp.shape[0], heads['write'].shape[0], attentions['write'].shape[0]
    l_w, r_w = inp.shape[1], inp2.shape[1]

    fig = plt.figure() 
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, head_h/float(in_h), attn_h/float(in_h)],
         width_ratios=[1,1,r_w/float(l_w),r_w/float(l_w)],
         wspace=0.05, hspace=0.00)
    ax1 = plt.subplot(gs[0,0])
    ax1.imshow(inp,interpolation='nearest', cmap='gray')
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax2 = plt.subplot(gs[0,1], sharey=ax1)
    ax2.imshow(tgt, interpolation='nearest', cmap='gray')
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax1b = plt.subplot(gs[0,2], sharey=ax2)
    ax1b.imshow(inp2,interpolation='nearest', cmap='gray')
    ax1b.set_xticklabels([])
    ax1b.set_yticklabels([])
    ax2b = plt.subplot(gs[0,3], sharey=ax1b)
    ax2b.imshow(tgt2, interpolation='nearest', cmap='gray')
    ax2b.set_xticklabels([])
    ax2b.set_yticklabels([])

    ax3 = plt.subplot(gs[1,0], sharex=ax1)
    ax3.imshow(heads['write'],interpolation='nearest')
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax4 = plt.subplot(gs[1,1], sharex=ax2, sharey=ax3)
    ax4.imshow(heads['read'],interpolation='nearest')
    ax4.set_xticklabels([])
    ax4.set_yticklabels([])
    ax3b = plt.subplot(gs[1,2], sharex=ax1b, sharey=ax4)
    ax3b.imshow(heads2['write'],interpolation='nearest')
    ax3b.set_xticklabels([])
    ax3b.set_yticklabels([])
    ax4b = plt.subplot(gs[1,3], sharex=ax2b, sharey=ax3b)
    ax4b.imshow(heads2['read'],interpolation='nearest')
    ax4b.set_xticklabels([])
    ax4b.set_yticklabels([])

    ax5 = plt.subplot(gs[2,0], sharex=ax3)
    ax5.imshow(attentions['write'], interpolation='nearest', cmap='gray')
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])
    ax6 = plt.subplot(gs[2,1], sharex=ax4, sharey=ax5)
    ax6.imshow(attentions['read'], interpolation='nearest', cmap='gray')
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])
    ax5b = plt.subplot(gs[2,2], sharex=ax3b, sharey=ax6)
    ax5b.imshow(attentions2['write'], interpolation='nearest', cmap='gray')
    ax5b.set_xticklabels([])
    ax5b.set_yticklabels([])
    ax6b = plt.subplot(gs[2,3], sharex=ax4b, sharey=ax5b)
    ax6b.imshow(attentions2['read'], interpolation='nearest', cmap='gray')
    ax6b.set_xticklabels([])
    ax6b.set_yticklabels([])

    plt.savefig(out_path)
    plt.clf()
    print('Image {} is saved.'.format(out_path))