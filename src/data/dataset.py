"""
CIFAR-10 Dataset Loading and Preprocessing
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    """
    Get CIFAR-10 train and test data loaders with appropriate preprocessing
    
    Args:
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of worker processes for data loading
        data_dir (str): Directory to store/load CIFAR-10 data
    
    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Normalization for testing (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    # CIFAR-10 class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')

    print(f"CIFAR-10 Dataset loaded:")
    print(f"  Training samples: {len(trainset)}")
    print(f"  Test samples: {len(testset)}")
    print(f"  Classes: {len(classes)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, test_loader, classes


def get_sample_batch(data_loader):
    """Get a sample batch for visualization"""
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    return images, labels


if __name__ == "__main__":
    # Test the data loader
    train_loader, test_loader, classes = get_cifar10_loaders(batch_size=4)
    
    # Get a sample batch
    images, labels = get_sample_batch(train_loader)
    print(f"Sample batch shape: {images.shape}")
    print(f"Sample labels: {labels}")
    print(f"Label names: {[classes[label] for label in labels]}")