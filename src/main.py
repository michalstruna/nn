import numpy as np
from ai import Perceptron
from matplotlib import pyplot as plt
from random import randrange

data = []

for i in range(1000):
	x1 = randrange(140, 220)
	x2 = randrange(40, 110)
	y = 1 if ((x1 + x2) < 250) else 0
	data.append([x1, x2, y])

data = np.array(data, dtype=float)

inputs = data[:, 0:data.shape[1] - 1]
expected = data[:, data.shape[1] - 1:]



nn = Perceptron()
training = nn.fit(inputs, expected)

print(training["error_history"])

plt.plot(training["error_history"])
plt.grid()
plt.show()