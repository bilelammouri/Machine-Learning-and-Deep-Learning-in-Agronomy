import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, TensorDataset
from torchviz import make_dot

class NeuralNetworkClassifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(NeuralNetworkClassifier, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def train_model(self, X, y, epochs=100, lr=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def plot_decision_boundary(self, X, y, title):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', marker='o', s=100, label='Original data')
        xx, yy = torch.meshgrid(torch.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100),
                                torch.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 100))
        Z = self(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1))
        Z = torch.argmax(Z, dim=1).reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
        plt.title(f'{title} Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()



# Example:
# Set random seed for reproducibility
torch.manual_seed(42)

# Generate synthetic data for binary classification
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Create an instance of NeuralNetworkClassifier with two hidden layers
model_two_hidden_layers = NeuralNetworkClassifier(input_size=2, output_size=2, hidden_layers=[10, 5])

# Create an instance of NeuralNetworkClassifier with three hidden layers
model_three_hidden_layers = NeuralNetworkClassifier(input_size=2, output_size=2, hidden_layers=[10, 8, 5])

# Train the models
model_two_hidden_layers.train_model(X, y)
model_three_hidden_layers.train_model(X, y)

# Plot the model architectures
make_dot(model_two_hidden_layers(X), params=dict(model_two_hidden_layers.named_parameters()))
make_dot(model_three_hidden_layers(X), params=dict(model_three_hidden_layers.named_parameters()))

# Plot decision boundaries using subplot
plt.figure(figsize=(12, 5))

# Plot decision boundaries for two hidden layers
plt.subplot(1, 2, 1)
model_two_hidden_layers.plot_decision_boundary(X, y, "Two Hidden Layers")

# Plot decision boundaries for three hidden layers
plt.subplot(1, 2, 2)
model_three_hidden_layers.plot_decision_boundary(X, y, "Three Hidden Layers")

plt.show()