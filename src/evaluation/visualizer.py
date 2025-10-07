"""
Visualization Functions for Model Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from matplotlib.patches import Rectangle


def plot_training_curves(history_resnet, history_plain, save_path=None):
    """Plot training curves comparison between ResNet and Plain CNN"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs_resnet = range(1, len(history_resnet['train_losses']) + 1)
    epochs_plain = range(1, len(history_plain['train_losses']) + 1)
    
    # Training Loss
    axes[0, 0].plot(epochs_resnet, history_resnet['train_losses'], 
                    label='ResNet', color='blue', linewidth=2)
    axes[0, 0].plot(epochs_plain, history_plain['train_losses'], 
                    label='Plain CNN', color='red', linewidth=2)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Test Accuracy
    axes[0, 1].plot(epochs_resnet, history_resnet['test_accuracies'], 
                    label='ResNet', color='blue', linewidth=2)
    axes[0, 1].plot(epochs_plain, history_plain['test_accuracies'], 
                    label='Plain CNN', color='red', linewidth=2)
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Accuracy
    axes[1, 0].plot(epochs_resnet, history_resnet['train_accuracies'], 
                    label='ResNet', color='blue', linewidth=2)
    axes[1, 0].plot(epochs_plain, history_plain['train_accuracies'], 
                    label='Plain CNN', color='red', linewidth=2)
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Gradient Norms
    axes[1, 1].plot(epochs_resnet, history_resnet['gradient_norms'], 
                    label='ResNet', color='blue', linewidth=2)
    axes[1, 1].plot(epochs_plain, history_plain['gradient_norms'], 
                    label='Plain CNN', color='red', linewidth=2)
    axes[1, 1].set_title('Gradient Norms')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Gradient Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_gradient_flow(model, input_tensor, save_path=None):
    """Visualize gradient flow through the network"""
    model.eval()
    
    # Forward pass
    output = model(input_tensor)
    
    # Backward pass with dummy loss
    dummy_loss = output.mean()
    dummy_loss.backward()
    
    # Collect gradient norms for each layer
    layer_names = []
    gradient_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None and 'weight' in name:
            layer_names.append(name.replace('.weight', ''))
            gradient_norms.append(param.grad.norm().item())
    
    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(gradient_norms)), gradient_norms)
    plt.xlabel('Layer')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Flow Through Network Layers')
    plt.xticks(range(len(layer_names)), layer_names, rotation=45, ha='right')
    plt.yscale('log')
    
    # Color bars based on gradient magnitude
    for i, bar in enumerate(bars):
        if gradient_norms[i] < 1e-4:
            bar.set_color('red')  # Vanishing gradients
        elif gradient_norms[i] > 1:
            bar.set_color('orange')  # Large gradients
        else:
            bar.set_color('green')  # Normal gradients
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_features(model, data_loader, layer_name='layer1.0.conv1', save_path=None):
    """Visualize feature maps from a specific layer"""
    model.eval()
    
    # Hook to capture feature maps
    features = []
    def hook(module, input, output):
        features.append(output.detach())
    
    # Register hook
    for name, module in model.named_modules():
        if name == layer_name:
            module.register_forward_hook(hook)
            break
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    
    # Forward pass
    with torch.no_grad():
        _ = model(images[:1])  # Use only first image
    
    if features:
        feature_map = features[0][0]  # First image, all channels
        
        # Plot feature maps
        num_channels = min(16, feature_map.shape[0])  # Show max 16 channels
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        
        for i in range(num_channels):
            row, col = i // 4, i % 4
            axes[row, col].imshow(feature_map[i].cpu(), cmap='viridis')
            axes[row, col].set_title(f'Channel {i}')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_channels, 16):
            row, col = i // 4, i % 4
            axes[row, col].axis('off')
        
        plt.suptitle(f'Feature Maps from {layer_name}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_model_comparison(resnet_results, plain_results, save_path=None):
    """Create comprehensive comparison plot"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Accuracy comparison
    models = ['ResNet-18', 'Plain CNN-18']
    accuracies = [resnet_results['accuracy'], plain_results['accuracy']]
    top5_accuracies = [resnet_results['top5_accuracy'], plain_results['top5_accuracy']]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, accuracies, width, label='Top-1 Accuracy', alpha=0.8)
    axes[0, 0].bar(x + width/2, top5_accuracies, width, label='Top-5 Accuracy', alpha=0.8)
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss comparison
    losses = [resnet_results['loss'], plain_results['loss']]
    axes[0, 1].bar(models, losses, color=['blue', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('Test Loss')
    axes[0, 1].set_title('Model Loss Comparison')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Per-class accuracy comparison (first 6 classes)
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog']
    resnet_class_acc = [resnet_results['classification_report'][str(i)]['precision'] 
                       for i in range(6)]
    plain_class_acc = [plain_results['classification_report'][str(i)]['precision'] 
                      for i in range(6)]
    
    x = np.arange(len(classes))
    axes[0, 2].bar(x - width/2, resnet_class_acc, width, label='ResNet', alpha=0.8)
    axes[0, 2].bar(x + width/2, plain_class_acc, width, label='Plain CNN', alpha=0.8)
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].set_title('Per-Class Precision (First 6 Classes)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(classes, rotation=45)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Confusion matrices
    sns.heatmap(resnet_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[1, 0], cbar=False)
    axes[1, 0].set_title('ResNet Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    sns.heatmap(plain_results['confusion_matrix'], annot=True, fmt='d', 
                cmap='Reds', ax=axes[1, 1], cbar=False)
    axes[1, 1].set_title('Plain CNN Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    # Model architecture comparison (simplified)
    axes[1, 2].text(0.1, 0.8, 'ResNet-18:', fontsize=14, fontweight='bold')
    axes[1, 2].text(0.1, 0.7, '• Skip connections', fontsize=12)
    axes[1, 2].text(0.1, 0.6, '• Better gradient flow', fontsize=12)
    axes[1, 2].text(0.1, 0.5, '• Deeper training possible', fontsize=12)
    
    axes[1, 2].text(0.1, 0.3, 'Plain CNN-18:', fontsize=14, fontweight='bold')
    axes[1, 2].text(0.1, 0.2, '• No skip connections', fontsize=12)
    axes[1, 2].text(0.1, 0.1, '• Gradient degradation', fontsize=12)
    axes[1, 2].text(0.1, 0.0, '• Training difficulties', fontsize=12)
    
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Architecture Differences')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Visualization module loaded successfully!")
    
    # Example usage would require actual training history and results
    # This is just for testing imports