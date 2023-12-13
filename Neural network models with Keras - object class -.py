import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
from keras.utils import to_categorical

# Define a class for a neural network classifier
class NeuralNetworkClassifier:
    def __init__(self, input_dim, num_classes, hidden_layers=2, neurons_per_layer=10):
        # Initialize the neural network with specified parameters
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.model = self.build_model()

    def build_model(self):
        # Build a neural network model for classification with customizable hidden layers
        model = Sequential()
        model.add(Dense(self.neurons_per_layer, input_dim=self.input_dim, activation='relu'))

        for _ in range(self.hidden_layers - 1):
            model.add(Dense(self.neurons_per_layer, activation='relu'))

        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X, y_one_hot, epochs=100, verbose=0):
        # Train the model
        self.model.fit(X, y_one_hot, epochs=epochs, verbose=verbose)

    def predict_decision_boundary(self, x_values, y_values):
        # Predict the decision boundary for visualization
        xx, yy = np.meshgrid(x_values, y_values)
        input_data = np.c_[xx.ravel(), yy.ravel()]
        Z = self.model.predict(input_data)
        Z = np.argmax(Z, axis=1).reshape(xx.shape)
        return xx, yy, Z

    def plot_decision_boundary(self, X, y, x_values, y_values):
        # Plot the original data and decision boundary
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', marker='o', s=100, label='Original data')

        xx, yy, Z = self.predict_decision_boundary(x_values, y_values)

        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
        plt.title('Neural Network Model for Classification')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()

# Define the main function
def main(X, y):
    # One-hot encode the labels
    y_one_hot = to_categorical(y, num_classes=2)

    # Create a NeuralNetworkClassifier instance with customized hidden layers
    neural_network = NeuralNetworkClassifier(input_dim=2, num_classes=2, hidden_layers=3, neurons_per_layer=8)

    # Plot the model architecture
    plot_model(neural_network.model, to_file='keras_classification_model.png', show_shapes=True, show_layer_names=True)

    # Train the model
    neural_network.train(X, y_one_hot)

    # Plot the decision boundary
    x_values = np.linspace(0, 1.5, 100)
    y_values = np.linspace(0, 1.5, 100)
    neural_network.plot_decision_boundary(X, y, x_values, y_values)

# Execute the main function when the script is run
if __name__ == "__main__":
    # Generate some example data for classification
    np.random.seed(42)
    X_example = np.random.rand(100, 2)
    y_example = (X_example[:, 0] + X_example[:, 1] > 1).astype(int)  # Simple binary classification task
    main(X_example, y_example)