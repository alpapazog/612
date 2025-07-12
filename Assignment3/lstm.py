import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

class AlphabetLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, output_size=26):
        super(AlphabetLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last time step
        return self.softmax(out)

alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_int = {c: i for i, c in enumerate(alphabet)}
int_to_char = {i: c for i, c in enumerate(alphabet)}

# Prepare X and Y as sequences of character indices
dataX = []
dataY = []

for i in range(len(alphabet) - 1):  # go until Y
    input_char = alphabet[i]
    output_char = alphabet[i + 1]
    dataX.append([char_to_int[input_char]])
    dataY.append(char_to_int[output_char])

# Convert to numpy arrays
dataX = np.array(dataX).reshape(-1, 1, 1)
dataY = np.array(dataY).reshape(-1, 1)

scaler = MinMaxScaler()
dataX = scaler.fit_transform(dataX.reshape(-1, 1)).reshape(-1, 1, 1)

encoder = OneHotEncoder(sparse_output=False, categories=[list(range(26))])
dataY = encoder.fit_transform([[char_to_int[alphabet[i + 1]]] for i in range(len(alphabet) - 1)])

# Convert data to PyTorch tensors
X_tensor = torch.tensor(dataX, dtype=torch.float32)  # shape: [25, 1, 1]
y_tensor = torch.tensor(dataY, dtype=torch.float32)  # shape: [25, 26]

model = AlphabetLSTM(num_layers=2)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Train loop
for epoch in range(100):
    model.train()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"[2-layer] Epoch {epoch+1}/100, Loss: {loss.item():.6f}")

# Accuracy
model.eval()
with torch.no_grad():
    preds = model(X_tensor)
    predicted_indices = torch.argmax(preds, dim=1)
    true_indices = torch.argmax(y_tensor, dim=1)
    acc = (predicted_indices == true_indices).sum().item() / len(true_indices)

print(f"[2-layer] Accuracy: {acc * 100:.2f}%")

print("\nSample Predictions:")
for i in range(len(alphabet)-1):
    # Invert normalization to get original input character
    input_val = scaler.inverse_transform([[dataX[i][0][0]]])[0][0]
    input_char = int_to_char[int(round(input_val))]

    # Get predicted character
    predicted_index = predicted_indices[i].item()
    predicted_char = int_to_char[predicted_index]

    print(f"Input: {input_char} -> Predicted: {predicted_char}")

layer_results = {}

for num_layers in [1, 2, 3, 4]:
    print(f"\nTraining with {num_layers} LSTM layer(s)...")

    model = AlphabetLSTM(num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        model.train()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        predicted_indices = torch.argmax(preds, dim=1)
        true_indices = torch.argmax(y_tensor, dim=1)
        acc = (predicted_indices == true_indices).sum().item() / len(true_indices)

    layer_results[num_layers] = {'loss': loss.item(), 'accuracy': acc}
    print(f"[{num_layers} layer(s)] Final Loss: {loss.item():.6f}, Accuracy: {acc * 100:.2f}%")

# This will print out performance for 48 total runs (3 layers × 4 hidden sizes × 4 learning rates).
hidden_sizes = [16, 32, 64, 128]
learning_rates = [0.1, 0.01, 0.001, 0.0001]
num_layers_list = [1, 2, 3]

print("\nTraining with different hyperparameters")
for num_layers in num_layers_list:
    for h in hidden_sizes:
        for lr in learning_rates:
            model = AlphabetLSTM(hidden_size=h, num_layers=num_layers)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            for epoch in range(100):
                model.train()
                output = model(X_tensor)
                loss = criterion(output, y_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_tensor)
                pred_indices = torch.argmax(preds, dim=1)
                true_indices = torch.argmax(y_tensor, dim=1)
                acc = (pred_indices == true_indices).sum().item() / len(true_indices)

            print(f"Layers: {num_layers}, Hidden: {h}, LR: {lr:.4f} -> Loss: {loss.item():.6f}, Acc: {acc * 100:.2f}%")