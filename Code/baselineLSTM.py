import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

class LSTMBaseline(nn.Module):

	def __init__(self, input_size, hidden_size, num_layers=1):
		super(LSTMBaseline, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
		self.out = nn.Linear(self.hidden_size, self.input_size)
		self.sigmoid = nn.Sigmoid()

		self.loss = nn.BCELoss()

	def forward(self, inputs):
		batch_size = inputs.size()[1]

		inputs = inputs.type(torch.FloatTensor)

		h_0 = c_0 = Variable(torch.zeros(1, batch_size,  self.hidden_size))

		outputs, hn = self.rnn(inputs, (h_0, c_0))
		return self.sigmoid(self.out(outputs))