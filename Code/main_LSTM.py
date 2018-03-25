import random
from random import randint
from data_gen import CopyTaskGenerator
import pdb
from baselineLSTM import *

if __name__ == '__main__':
	data_gen = CopyTaskGenerator()
	inp, target = data_gen.generate(100, 20)
	inp = inp[:-1,:,:-1]

	baseline_model = LSTMBaseline(inp.size(2), target.size(2))
	y_out = baseline_model(inp)

	pdb.set_trace()

	loss = baseline_model.loss(y_out, target)


	pdb.set_trace()