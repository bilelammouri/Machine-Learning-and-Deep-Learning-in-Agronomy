# Import the NumPy library
import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights with random values
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def forward(self, X):
        # Forward pass through the network
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden)
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)

        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        self.predicted_output = sigmoid(self.output_layer_input)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Backpropagation to update weights
        error = y - self.predicted_output

        # Calculate gradients
        output_error = error * sigmoid_derivative(self.predicted_output)
        hidden_layer_error = output_error.dot(self.weights_hidden_output.T) * sigmoid_derivative(self.hidden_layer_output)

        # Update weights
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_error) * learning_rate
        self.weights_input_hidden += X.T.dot(hidden_layer_error) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward and backward pass for each epoch
            output = self.forward(X)
            self.backward(X, y, learning_rate)

            # Print the mean squared error for every 1000 epochs
            if epoch % 1000 == 0:
                mse = np.mean((y - output) ** 2)
                print(f"Epoch {epoch}, Mean Squared Error: {mse}")


# Example usage:
# Create a dataset for binary classification
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create a neural network with 2 input neurons, 2 hidden neurons, and 1 output neuron
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)

# Train the neural network
nn.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained neural network
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = nn.forward(test_input)
print("Predictions:")
print(predictions)