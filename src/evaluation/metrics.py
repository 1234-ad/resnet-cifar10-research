"""
Evaluation Metrics and Model Analysis
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, test_loader, device='cuda', num_classes=10):
    """
    Comprehensive model evaluation
    
    Returns:
        dict: Dictionary containing various evaluation metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(test_loader)
    
    # Per-class metrics
    class_report = classification_report(all_targets, all_predictions, 
                                       output_dict=True, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    # Top-k accuracy
    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)
    
    top5_correct = 0
    for i in range(len(all_targets)):
        top5_pred = np.argsort(all_probabilities[i])[-5:]
        if all_targets[i] in top5_pred:
            top5_correct += 1
    
    top5_accuracy = 100. * top5_correct / len(all_targets)
    
    results = {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'loss': avg_loss,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }
    
    return results


def calculate_metrics(results):
    """Calculate additional metrics from evaluation results"""
    
    # Extract per-class metrics
    class_report = results['classification_report']
    
    # Macro and weighted averages
    macro_precision = class_report['macro avg']['precision']
    macro_recall = class_report['macro avg']['recall']
    macro_f1 = class_report['macro avg']['f1-score']
    
    weighted_precision = class_report['weighted avg']['precision']
    weighted_recall = class_report['weighted avg']['recall']
    weighted_f1 = class_report['weighted avg']['f1-score']
    
    # Per-class accuracy
    conf_matrix = results['confusion_matrix']
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    metrics = {
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_accuracy': per_class_acc,
        'mean_per_class_accuracy': np.mean(per_class_acc)
    }
    
    return metrics


def plot_confusion_matrix(conf_matrix, classes, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_model_complexity(model):
    """Analyze model complexity and architecture"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate FLOPs (approximate for CNN)
    def count_conv_flops(layer, input_shape):
        if isinstance(layer, nn.Conv2d):
            output_dims = input_shape[2] // layer.stride[0]
            kernel_flops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
            output_elements = layer.out_channels * output_dims * output_dims
            return kernel_flops * output_elements
        return 0
    
    # This is a simplified FLOP calculation
    # For more accurate calculation, use tools like ptflops or fvcore
    
    complexity_info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
    }
    
    return complexity_info


if __name__ == "__main__":
    # Test evaluation functions
    print("Evaluation metrics module loaded successfully!")
    
    # Example usage would require actual model and data
    # This is just for testing imports