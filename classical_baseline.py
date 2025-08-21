import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data_setup import load_data

# --- Hyperparameters ---
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

class ClassicalModel(nn.Module):
    """A simple classical neural network for baseline comparison."""
    def __init__(self):
        super(ClassicalModel, self).__init__()
        # A simple MLP structure
        # The number of parameters should be roughly comparable to the hybrid model
        self.layer1 = nn.Linear(64, 16)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 2) # 2 classes (0, 1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return self.log_softmax(x)

def train_classical_model():
    """Main function to train and evaluate the classical baseline model."""
    # --- 1. Load Data ---
    print("Loading and preparing data for classical model...")
    x_train, x_test, y_train, y_test = load_data()

    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 2. Initialize Model, Optimizer, and Loss ---
    print("Initializing classical model...")
    model = ClassicalModel()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.NLLLoss()

    # --- 3. Training Loop ---
    print("Starting classical training...")
    loss_history = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    print("Finished Classical Training.")

    # --- 4. Evaluation ---
    print("Evaluating classical model...")
    with torch.no_grad():
        test_outputs = model(x_test)
        _, predicted = torch.max(test_outputs, 1)
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        print(f'Accuracy of the classical model on the test set: {100 * accuracy:.2f}%')

    # --- 5. Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Classical Model Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("classical_training_loss.png")
    print("Classical training loss plot saved to classical_training_loss.png")

if __name__ == '__main__':
    train_classical_model()
