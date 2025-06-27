import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Load and prepare data
df = pd.read_csv("housing.csv", header=None, delim_whitespace=True)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

# Device is cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Custom model with one hidden layer
class SimpleFCNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleFCNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.out(x)

# Hyperparams
hidden_sizes = [4, 8, 16, 32, 64]
epochs = 100
batch_size = 16
k = 5

# Store results
results = {}

for hidden_dim in hidden_sizes:
    print(f"\nTraining model with {hidden_dim} hidden units...")
    mse_scores = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        # Split & scale
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Model
        model = SimpleFCNet(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            mse = mean_squared_error(
                y_test_tensor.cpu().numpy(),
                predictions.cpu().numpy()
            )
            mse_scores.append(mse)

    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    results[hidden_dim] = (mean_mse, std_mse)
    print(f"Mean MSE: {mean_mse:.4f}, Std: {std_mse:.4f}")
