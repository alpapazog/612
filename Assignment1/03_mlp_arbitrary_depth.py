import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv("housing.csv", header=None, delim_whitespace=True)
X = df.iloc[:, :-1].values.astype(np.float32)
y = df.iloc[:, -1].values.astype(np.float32).reshape(-1, 1)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# General FC network
class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(FullyConnectedNet, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Configs to test
layer_configs = {
    "1 layer": [32],
    "2 layers": [64, 32],
    "3 layers": [128, 64, 32],
    "4 layers": [128, 64, 32, 16],
    "5 layers": [256, 128, 64, 32, 16],
}

# Hyperparameters
epochs = 100
batch_size = 16
k = 5

# Store results
results = {}
depth_loss_log = {}  # store average loss per epoch for each depth

for label, hidden_dims in layer_configs.items():
    print(f"\nTraining {label} model...")
    mse_scores = []
    all_fold_losses = []  # store losses from all folds

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        # Split + scale
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Tensors to device
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                  batch_size=batch_size, shuffle=True)

        # Model, loss, optimizer
        model = FullyConnectedNet(input_dim=13, hidden_dims=hidden_dims).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        epoch_losses = []  # track per-epoch average loss
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(train_loader)
            epoch_losses.append(avg_loss)
        all_fold_losses.append(epoch_losses)

        # Evaluation
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor)
            mse = mean_squared_error(y_test_tensor.cpu().numpy(), predictions.cpu().numpy())
            mse_scores.append(mse)

    # Average loss across folds per epoch
    depth_loss_log[label] = np.mean(all_fold_losses, axis=0).tolist()

    # Log results
    mean_mse = np.mean(mse_scores)
    std_mse = np.std(mse_scores)
    results[label] = (mean_mse, std_mse)
    print(f"{label} → Mean MSE: {mean_mse:.4f}, Std: {std_mse:.4f}")

# Print summary
print("\nSummary of all results:")
for label, (mean_mse, std_mse) in results.items():
    print(f"{label:10s} → Mean MSE: {mean_mse:.4f}, Std: {std_mse:.4f}")

# Plot convergence curves
plt.figure(figsize=(10, 6))
for label, losses in depth_loss_log.items():
    plt.plot(losses, label=label)
plt.xlabel("Epoch")
plt.ylabel("Average Training Loss")
plt.title("Convergence Speed by Network Depth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()