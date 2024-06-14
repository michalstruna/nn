import numpy as np

class Perceptron:

	def __init__(self, input_shape):
		self.input_shape = input_shape
		self.weights = np.random.rand(input_shape[0])
		self.bias = np.random.rand(1)

	def train(self, inputs, expected, epochs=100, rate=0.05, test=0.25):
		error_history = []
		split_index = round(inputs.shape[0] * test)

		train_inputs = inputs[:split_index]
		test_inputs = inputs[split_index:]

		train_expected = expected[:split_index]
		test_expected = expected[split_index:]

		for _ in range(epochs):
			epoch_error_history = []

			for i in range(len(train_inputs)):
				weighted_sum = self._sum(train_inputs[i], self.weights)
				result = self._activation(weighted_sum)
				error = np.sum(result - train_expected[i])
				self.weights -= error * rate
				self.bias -= error * rate
				epoch_error_history.append(np.abs(error))

			error_history.append(np.mean(epoch_error_history))

		results = test_expected.copy()

		for i in range(len(test_inputs)):
			weighted_sum = self._sum(test_inputs[i], self.weights)
			results[i][0] = self._activation(weighted_sum)

		return {
			"results": results,
			"expected": test_expected,
			"error_history": error_history,
			"test_error": np.sum(np.abs(test_expected - results))
		}

	def _sum(self, x, w):
		return np.sum(x * w) - self.bias

	def _activation(self, x):
		return 1 if x > 0 else 0