import torch
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

def load_data(dataset_name='MNIST', test_split=0.2, random_seed=42):
    """
    Loads, preprocesses, and splits the specified dataset.

    Args:
        dataset_name (str): The name of the dataset to load ('MNIST' or 'FashionMNIST').
        test_split (float): The proportion of the dataset to reserve for testing.
        random_seed (int): Seed for reproducibility.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        x_train, x_test, y_train, y_test
    """
    # Transformations to apply to the images
    transform = transforms.Compose([
        transforms.Resize((8, 8)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)), # Normalize to [-1, 1]
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    # Download and transform the training and test data
    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name == 'FashionMNIST':
        train_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        raise ValueError("Dataset not supported. Please choose 'MNIST' or 'FashionMNIST'.")

    # Combine datasets
    full_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    # Extract data and labels
    X = torch.stack([img for img, label in full_dataset])
    y = torch.tensor([label for img, label in full_dataset])

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=random_seed, stratify=y
    )

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    # Example of how to use the function
    x_train, x_test, y_train, y_test = load_data()
    print("Data loading complete.")
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print(f"Number of training samples: {len(y_train)}")
    print(f"Number of test samples: {len(y_test)}")
    print(f"Sample training label: {y_train[0]}")
    print(f"Sample training image (first 10 features):\n{x_train[0][:10]}")
