"""
Main Training Script for ResNet vs Plain CNN Comparison
"""

import argparse
import torch
import os
from datetime import datetime

from src.models import ResNet18, PlainCNN18
from src.data import get_cifar10_loaders
from src.training import Trainer, set_seed, count_parameters
from src.evaluation import evaluate_model, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description='ResNet vs Plain CNN Training')
    parser.add_argument('--model', type=str, choices=['resnet', 'plain'], 
                       required=True, help='Model type to train')
    parser.add_argument('--epochs', type=int, default=200, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, 
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    parser.add_argument('--data_dir', type=str, default='./data', 
                       help='Data directory')
    parser.add_argument('--save_dir', type=str, default='./results', 
                       help='Save directory for results')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto, cuda, cpu)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create save directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join(args.save_dir, f"{args.model}_{timestamp}")
    os.makedirs(model_save_dir, exist_ok=True)
    
    log_dir = os.path.join(model_save_dir, 'logs')
    checkpoint_dir = os.path.join(model_save_dir, 'checkpoints')
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, classes = get_cifar10_loaders(
        batch_size=args.batch_size, data_dir=args.data_dir
    )
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'resnet':
        model = ResNet18(num_classes=10)
        print("ResNet-18 with skip connections")
    else:
        model = PlainCNN18(num_classes=10)
        print("Plain CNN-18 without skip connections")
    
    # Print model info
    count_parameters(model)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        log_dir=log_dir
    )
    
    # Train model
    print(f"Starting training for {args.epochs} epochs...")
    best_acc = trainer.train(epochs=args.epochs, save_path=checkpoint_dir)
    
    # Final evaluation
    print("Performing final evaluation...")
    results = evaluate_model(model, test_loader, device)
    
    # Save results
    import pickle
    results_path = os.path.join(model_save_dir, 'results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'model_type': args.model,
            'best_accuracy': best_acc,
            'final_results': results,
            'training_history': trainer.get_training_history(),
            'args': vars(args)
        }, f)
    
    print(f"Training completed!")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {model_save_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {args.model.upper()}")
    print(f"Final Test Accuracy: {results['accuracy']:.2f}%")
    print(f"Final Test Loss: {results['loss']:.4f}")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("="*50)


if __name__ == "__main__":
    main()