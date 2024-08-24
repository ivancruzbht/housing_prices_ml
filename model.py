import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.

    Args:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer.
        dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.5.

    Attributes:
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        fc3 (nn.Linear): The third fully connected layer.
        relu (nn.ReLU): The ReLU activation function.
        dropout (nn.Dropout): The dropout layer.

    Methods:
        forward(x): Performs forward pass through the MLP.

    Returns:
        torch.Tensor: The output tensor from the MLP.
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x) 
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x 

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    """
    PyTorchRegressor is a regression model implemented using PyTorch.
    Parameters:
    - model: The PyTorch model used for regression.
    - epochs: The number of training epochs (default: 100).
    - lr: The learning rate for the optimizer (default: 0.01).
    Methods:
    - fit(X, y): Fit the regression model to the training data.
    - predict(X): Make predictions on new data.
    Example usage:
    model = PyTorchRegressor(model=my_model, epochs=100, lr=0.01)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    """

    def __init__(self, model, epochs=100, learning_rate=0.01):
        self.model = model.to(device)
        self.epochs = epochs
        self.lr = learning_rate

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        for _ in tqdm(range(self.epochs), desc="Training Progress"):
            optimizer.zero_grad()
            outputs = self.model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X).cpu().numpy()
        return predictions.flatten()