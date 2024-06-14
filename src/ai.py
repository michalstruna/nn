import numpy as np

class Perceptron:

	def __init__(self):
		self.weights = None
		self.boundaries = None
	
	def predict(self, input):
		normalized = self._normalize(input)
		z = self._linear(normalized)
		return self._activation(z)
	
	def _split_inputs(self, inputs, targets, test_coeff):
		split_index = round(inputs.shape[0] * (1 - test_coeff))
		train_inputs, test_inputs = inputs[:split_index], inputs[split_index:]
		train_targets, test_targets = targets[:split_index], targets[split_index:]
		return train_inputs, train_targets, test_inputs, test_targets


	def fit(self, inputs, targets, epochs=100, rate=0.01, test=0.25):
		_, *input_shape = inputs.shape
		self._create_boundaries(inputs)

		if self.weights is None:
			self.weights = np.random.rand(input_shape[0] + 1)

		train_inputs, train_targets, test_inputs, test_targets = self._split_inputs(inputs, targets, test)
		error_history = []
		
		for _ in range(epochs):
			epoch_error_history = []

			for i in range(len(train_inputs)):
				error = self._train(train_inputs[i], train_targets[i], rate)
				epoch_error_history.append(np.abs(error))

			error_history.append(np.mean(epoch_error_history))

		return {
			"error_history": error_history
		}
	
	def _train(self, input, target, rate):
		prediction = self.predict(input)
		error = self._loss(prediction, target)
		self.weights[1:] -= rate * error * input
		self.weights[0] -= rate * error
		return error

	def _loss(self, prediction, target):
		return prediction - target
	
	def _linear(self, input):
		return input @ self.weights[1:].T + + self.weights[0]
	
	def _activation(self, z):
		return 1 if z >= 0 else 0
	
	def _create_boundaries(self, inputs):
		self.boundaries = []

		for i in range(inputs.shape[1]):
			minimum, maximum = inputs[:,i].min(), inputs[:,i].max()
			self.boundaries.append((minimum, maximum))
	
	def _normalize(self, input):
		result = input.copy()

		for i in range(len(self.boundaries)):
			minimum, maximum = self.boundaries[i]
			result[i] = (input[i] - minimum) / (maximum - minimum)

		return result