import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

# Define the quantum device
# n_qubits = 6 because we will compress the 64 pixels to 6 features
n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_circuit(inputs, weights):
    """The variational quantum circuit.

    Args:
        inputs (torch.Tensor): Input features (6 per image).
        weights (torch.Tensor): Trainable parameters/weights (3 layers, 6 qubits, 3 rotations each).
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

    # Return the expectation value of the Pauli-Z operator on the first qubit
    return qml.expval(qml.PauliZ(0))

class HybridModel(nn.Module):
    """Hybrid quantum-classical neural network.
    """
    def __init__(self):
        super(HybridModel, self).__init__()
        # Classical layer to reduce dimensions from 64 to 6
        self.classical_layer = nn.Linear(64, n_qubits)
        
        # Quantum layer
        self.quantum_weights = nn.Parameter(torch.randn(4, n_qubits))
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes={"weights": (4, n_qubits)})

        # Classical output layer
        self.output_layer = nn.Linear(1, 2) # 1 output from Q-layer, 2 classes (0, 1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Apply classical layer
        x = self.classical_layer(x)
        
        # Apply quantum layer
        # The TorchLayer expects the weights to be passed explicitly if they are nn.Parameters
        x = self.quantum_layer(x)

        # Apply output layer
        x = self.output_layer(x.reshape(-1, 1))
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
