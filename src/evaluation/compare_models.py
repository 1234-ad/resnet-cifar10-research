"""
Model Comparison Script
Run this after training both models to generate comparison plots and analysis
"""

import pickle
import os
import argparse
import matplotlib.pyplot as plt
from visualizer import plot_training_curves, plot_model_comparison


def load_results(results_dir):
    """Load training results from pickle files"""
    resnet_path = None
    plain_path = None
    
    # Find result files
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        if os.path.isdir(item_path):
            results_file = os.path.join(item_path, 'results.pkl')
            if os.path.exists(results_file):
                with open(results_file, 'rb') as f:
                    data = pickle.load(f)
                    if data['model_type'] == 'resnet':
                        resnet_path = results_file
                    elif data['model_type'] == 'plain':
                        plain_path = results_file
    
    if not resnet_path or not plain_path:
        raise FileNotFoundError("Could not find both ResNet and Plain CNN results")
    
    # Load data
    with open(resnet_path, 'rb') as f:
        resnet_data = pickle.load(f)
    
    with open(plain_path, 'rb') as f:
        plain_data = pickle.load(f)
    
    return resnet_data, plain_data


def main():
    parser = argparse.ArgumentParser(description='Compare ResNet vs Plain CNN results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing training results')
    parser.add_argument('--output_dir', type=str, default='./comparison_plots',
                       help='Directory to save comparison plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading training results...")
    resnet_data, plain_data = load_results(args.results_dir)
    
    # Extract data
    resnet_history = resnet_data['training_history']
    plain_history = plain_data['training_history']
    resnet_results = resnet_data['final_results']
    plain_results = plain_data['final_results']
    
    # Generate comparison plots
    print("Generating training curves comparison...")
    plot_training_curves(
        resnet_history, plain_history,
        save_path=os.path.join(args.output_dir, 'training_curves.png')
    )
    
    print("Generating model comparison...")
    plot_model_comparison(
        resnet_results, plain_results,
        save_path=os.path.join(args.output_dir, 'model_comparison.png')
    )
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"ResNet-18 Best Accuracy: {resnet_data['best_accuracy']:.2f}%")
    print(f"Plain CNN-18 Best Accuracy: {plain_data['best_accuracy']:.2f}%")
    print(f"Improvement: {resnet_data['best_accuracy'] - plain_data['best_accuracy']:.2f} percentage points")
    print(f"Final Test Accuracy - ResNet: {resnet_results['accuracy']:.2f}%")
    print(f"Final Test Accuracy - Plain: {plain_results['accuracy']:.2f}%")
    print(f"Final Test Loss - ResNet: {resnet_results['loss']:.4f}")
    print(f"Final Test Loss - Plain: {plain_results['loss']:.4f}")
    print("="*60)
    
    print(f"\nComparison plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()