# Quantum-Classical Neural Network for MNIST Classification

This project implements and compares a hybrid quantum-classical neural network with a purely classical neural network for classifying images of digits 0 and 1 from the MNIST dataset.

## Dataset

The project uses the MNIST dataset, which is automatically downloaded and preprocessed. The dataset is filtered to only include images of digits 0 and 1. The images are resized to 8x8 pixels and flattened into a 64-element vector.

## Models

### Hybrid Quantum-Classical Model

The hybrid model consists of three main parts:

1.  **Classical Layer:** A linear layer that reduces the dimensionality of the input image from 64 features to 6.
2.  **Quantum Layer:** A variational quantum circuit implemented using PennyLane. The 6 features from the classical layer are encoded as rotation angles in the quantum circuit. The circuit has trainable layers and an entanglement layer.
3.  **Classical Output Layer:** A final linear layer that takes the output of the quantum circuit and produces the final classification.

### Classical Baseline Model

A simple Multi-Layer Perceptron (MLP) is implemented as a baseline for comparison. The architecture is designed to have a roughly comparable number of parameters to the hybrid model.

## Files

-   `hybrid_model.py`: Defines the architecture of the hybrid quantum-classical model.
-   `classical_baseline.py`: Defines the architecture of the classical baseline model.
-   `data_setup.py`: Handles the loading, preprocessing, and splitting of the MNIST dataset.
-   `train.py`: Contains the training and evaluation loop for the hybrid model.
-   `requirements.txt`: Lists the necessary Python packages for this project.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Hybrid Model:**
    ```bash
    python train.py
    ```

3.  **Train the Classical Baseline Model:**
    ```bash
    python classical_baseline.py
    ```

## Outputs

Running the training scripts will produce the following files:

-   `training_loss.png`: A plot of the training loss over epochs for the hybrid model.
-   `classical_training_loss.png`: A plot of the training loss over epochs for the classical model.
