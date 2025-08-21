import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Define the quantum device
# n_qubits = 10 because we will compress the 64 pixels to 10 features
n_qubits = 10
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    """The variational quantum circuit.

    Args:
        inputs (torch.Tensor): Input features (10 per image).
        weights (torch.Tensor): Trainable parameters/weights.
    """
    # Encode input features as rotation angles
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # Trainable circuit layers
    for i in range(n_qubits):
        qml.RY(weights[0, i], wires=i)
        qml.RZ(weights[1, i], wires=i)

    # Entanglement layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    for i in range(n_qubits):
        qml.RY(weights[2, i], wires=i)
        qml.RZ(weights[3, i], wires=i)

    # Second Entanglement layer
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

    for i in range(n_qubits):
        qml.RY(weights[4, i], wires=i)
        qml.RZ(weights[5, i], wires=i)

    # Return the expectation value of the Pauli-Z operator on each qubit
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class HybridModel(nn.Module):
    """Hybrid quantum-classical neural network.
    """
    def __init__(self):
        super(HybridModel, self).__init__()
        # Classical layer to reduce dimensions from 64 to 10
        self.classical_layer = nn.Linear(64, n_qubits)
        
        # Quantum layer
        self.quantum_weights = nn.Parameter(torch.randn(6, n_qubits))
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={"weights": (6, n_qubits)})

        # Classical output layer
        self.output_layer = nn.Linear(n_qubits, 10) # 10 outputs from Q-layer, 10 classes (0-9)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Apply classical layer
        x = self.classical_layer(x)
        
        # Apply quantum layer
        x = self.quantum_layer(x)

        # Apply output layer
        x = self.output_layer(x)
        return self.log_softmax(x)

if __name__ == '__main__':
    # Test the model with a random input
    model = HybridModel()
    random_input = torch.randn(1, 64) # Batch size of 1, 64 features
    output = model(random_input)
    
    print("Hybrid model created successfully.")
    print(f"Input shape: {random_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (log probabilities): {output}")
    print(f"Predicted class: {torch.argmax(output, dim=1).item()}")
