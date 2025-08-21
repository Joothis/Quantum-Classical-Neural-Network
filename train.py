import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from data_setup import load_data
from hybrid_model import HybridModel

# --- Hyperparameters ---
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.01

def train_model():
    """Main function to train and evaluate the hybrid model."""
    # --- 1. Load Data ---
    print("Loading and preparing data...")
    x_train, x_test, y_train, y_test = load_data()

    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 2. Initialize Model, Optimizer, and Loss ---
    print("Initializing model...")
    model = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.NLLLoss() # Negative Log Likelihood Loss

    # --- 3. Training Loop ---
    print("Starting training...")
    loss_history = []
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    print("Finished Training.")

    # --- 4. Evaluation ---
    print("Evaluating model...")
    with torch.no_grad():
        # Get predictions for the test set
        test_outputs = model(x_test)
        _, predicted = torch.max(test_outputs, 1)
        
        # Calculate accuracy
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        print(f'Accuracy on the test set: {100 * accuracy:.2f}%')

    # --- 5. Plotting ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss.png")
    print("Training loss plot saved to training_loss.png")

if __name__ == '__main__':
    train_model()
