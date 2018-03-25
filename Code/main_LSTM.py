import random
from random import randint
from data_gen import CopyTaskGenerator
import pdb
from baselineLSTM import *
from torch import optim

if __name__ == '__main__':
	data_gen = CopyTaskGenerator()
	inp, target = data_gen.generate(100, 20)
	inp = inp[:-1,:,:-1]

	baseline_model = LSTMBaseline(inp.size(2), target.size(2))

	optimizer = optim.RMSprop(baseline_model.parameters(),
                             momentum=0.9,
                             alpha=0.95,
                             lr=1e-3)


	baseline_model.train()

	for i in range(200):
		optimizer.zero_grad()
		y_out = baseline_model(inp)

		loss = baseline_model.loss(y_out, target)
		loss.backward()
		optimizer.step()

		if i % 20:
			print(float(loss.data))
