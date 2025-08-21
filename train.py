import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse

from data_setup import load_data
from hybrid_model import HybridModel

# --- Hyperparameters ---
EPOCHS = 10

def train_model(learning_rate, batch_size, dataset_name, use_gpu):
    """Main function to train and evaluate the hybrid model."""
    # --- 1. Load Data ---
    x_train, x_test, y_train, y_test = load_data(dataset_name=dataset_name)

    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # --- 2. Initialize Model, Optimizer, and Loss ---
    model = HybridModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss() # Negative Log Likelihood Loss

    # --- 3. Training Loop ---
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
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

    # --- 4. Evaluation ---
    with torch.no_grad():
        # Get predictions for the test set
        test_outputs = model(x_test)
        _, predicted = torch.max(test_outputs, 1)
        
        # Calculate accuracy
        correct = (predicted == y_test).sum().item()
        total = y_test.size(0)
        accuracy = correct / total
        return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hybrid Quantum-Classical Neural Network Training')
    parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'FashionMNIST'],
                        help='Dataset to use for training (MNIST or FashionMNIST)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training if available')
    args = parser.parse_args()

    # --- Hyperparameter Tuning ---
    learning_rates = [0.01, 0.001, 0.0001]
    batch_sizes = [32, 64, 128]
    best_accuracy = 0
    best_hyperparameters = {}

    print(f"Starting hyperparameter tuning for {args.dataset} dataset...")
    for lr in learning_rates:
        for bs in batch_sizes:
            print(f"\nTraining with learning_rate={lr}, batch_size={bs}")
            accuracy = train_model(lr, bs, args.dataset, args.gpu)
            print(f"Accuracy: {100 * accuracy:.2f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_hyperparameters = {"learning_rate": lr, "batch_size": bs}

    print("\nFinished hyperparameter tuning.")
    print(f"Best accuracy: {100 * best_accuracy:.2f}%")
    print(f"Best hyperparameters: {best_hyperparameters}")