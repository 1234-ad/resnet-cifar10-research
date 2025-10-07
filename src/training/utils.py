"""
Training Utilities
"""

import torch
import os


def save_checkpoint(state, filename='checkpoint.pth'):
    """Save model checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    if os.path.isfile(filename):
        print(f"Loading checkpoint: {filename}")
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0)
        
        print(f"Loaded checkpoint from epoch {epoch}, best accuracy: {best_acc:.2f}%")
        return epoch, best_acc
    else:
        print(f"No checkpoint found at: {filename}")
        return 0, 0


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return total_params, trainable_params


def get_model_size(model):
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")


if __name__ == "__main__":
    # Test utilities
    from src.models import ResNet18, PlainCNN18
    
    resnet = ResNet18()
    plain_cnn = PlainCNN18()
    
    print("ResNet-18:")
    count_parameters(resnet)
    print(f"Model size: {get_model_size(resnet):.2f} MB")
    
    print("\nPlain CNN-18:")
    count_parameters(plain_cnn)
    print(f"Model size: {get_model_size(plain_cnn):.2f} MB")