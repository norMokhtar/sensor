# Importing Essential Libraries and Modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import dgl
from dgl.nn.pytorch import GraphConv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load Dataset from CSV File
drift_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_ph/drift_data_ph4_7_10_31082023.xls')

# Splitting the Data: 80% for Training and 20% for Testing
train_data, test_data = train_test_split(drift_data, test_size=0.2, random_state=42)

# Z-score Normalization Function
def normalize_data(data):
    return (data - data.mean()) / data.std()

# Preprocess Data for Training and Testing
def preprocess_data(data):
    features = data['sensor_read'].values
    features = normalize_data(features)
    features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)

    ground_truth = data['drift_error'].values
    ground_truth = normalize_data(ground_truth)
    ground_truth = torch.tensor(ground_truth, dtype=torch.float32).unsqueeze(1)

    return features, ground_truth

# Process Training and Testing Data
train_features, train_ground_truth = preprocess_data(train_data)
test_features, test_ground_truth = preprocess_data(test_data)

# Constructing a Graph Based on Sensor Readings Sequence
def construct_graph(data):
    num_nodes = len(data)
    edges = ([i for i in range(num_nodes-1)] + [num_nodes-1], [i+1 for i in range(num_nodes-1)] + [0])
    return dgl.graph(edges)

# Construct Graphs for Training and Testing Data
train_g = construct_graph(train_data)
test_g = construct_graph(test_data)

# Define the Graph Recurrent Neural Network (GRNN) Model
class GRNN(nn.Module):
    def __init__(self, in_feats, hidden_size1, hidden_size2, lstm_hidden, num_classes):
        super(GRNN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size1)  # First Graph Convolution Layer
        self.conv2 = GraphConv(hidden_size1, hidden_size2)  # Second Graph Convolution Layer
        self.lstm = nn.LSTM(hidden_size2, lstm_hidden, batch_first=True)  # LSTM Layer
        self.fc = nn.Linear(lstm_hidden, num_classes)  # Fully Connected Layer for Output

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features))  # First Graph Convolution with ReLU Activation
        x = F.relu(self.conv2(g, x))  # Second Graph Convolution with ReLU Activation
        x, _ = self.lstm(x.unsqueeze(0))  # LSTM Layer
        x = self.fc(x.squeeze(0))  # Fully Connected Layer
        return x

# Define Hyperparameters
input_size = 1
hidden_size1 = 64
hidden_size2 = 64
lstm_hidden = 32
output_size = 1
learning_rate = 0.001
num_epochs = 500

# Model Initialization
model = GRNN(input_size, hidden_size1, hidden_size2, lstm_hidden, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()

# List for Storing MSE Values During Training
mse_values = []

# Training Loop for the GRNN Model
for epoch in range(num_epochs):
    logits = model(train_g, train_features)  # Model Predictions for the Current Epoch
    loss = loss_function(logits, train_ground_truth)  # Loss Calculation
    mse_values.append(loss.item())

    # Print MSE every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, MSE: {loss.item():.4f}")

    optimizer.zero_grad()  # Reset Gradients
    loss.backward()  # Backpropagation
    optimizer.step()  # Model Parameter Update

# Evaluate on Test Data
test_predictions = model(test_g, test_features).detach()

# Display Comparison of Actual vs. Predicted Values on Test Data
print("\nComparing Actual vs. Predicted Drift Errors on Test Data (first few values):")
print("\n{:<15} {:<15}".format('Actual', 'Predicted'))
print("-" * 30)
for actual, predicted in zip(test_ground_truth[:10], test_predictions[:10]):
    print("{:<15.6f} {:<15.6f}".format(actual.item(), predicted.item()))

# Visualization: Actual vs. Predicted Drift Errors and MSE during Training
plt.figure(figsize=(12, 6))

# Actual vs. Predicted Drift Errors
plt.subplot(1, 2, 1)
plt.plot(train_ground_truth, label="Actual Drift Error", color="blue")
plt.plot(model(train_g, train_features).detach(), label="Predicted Drift Error", color="red", linestyle="--")
plt.xlabel("Data Points")
plt.ylabel("Drift Error Value")
plt.title("Actual vs. Predicted Drift Errors (Training Data)")
plt.legend()
plt.grid(True)

# MSE During Training
plt.subplot(1, 2, 2)
plt.plot(mse_values, label="MSE", color="green")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.title("Training Progress - Mean Squared Error")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
