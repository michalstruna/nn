import numpy as np
from ai import Perceptron
from matplotlib import pyplot as plt
from random import randrange

data = []

for i in range(100):
	x1 = randrange(140, 210)
	y = 1 if x1 < 170 else 0
	data.append([x1, y])

data = np.array(data)

inputs = data[:, 0:data.shape[1] - 1]
expected = data[:, data.shape[1] - 1:]
_, *input_shape = inputs.shape


nn = Perceptron(input_shape=input_shape)
training = nn.train(inputs=inputs,expected=expected)

print(training["error_history"])

plt.plot(training["error_history"])
plt.grid()
plt.show()