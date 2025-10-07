# Usage Guide: ResNet vs Plain CNN on CIFAR-10

This guide provides step-by-step instructions for running the ResNet vs Plain CNN comparison experiments.

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/1234-ad/resnet-cifar10-research.git
cd resnet-cifar10-research

# Create virtual environment (recommended)
python -m venv resnet_env
source resnet_env/bin/activate  # On Windows: resnet_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Training Models

#### Train ResNet-18
```bash
python train.py --model resnet --epochs 200 --batch_size 128 --lr 0.1
```

#### Train Plain CNN-18
```bash
python train.py --model plain --epochs 200 --batch_size 128 --lr 0.1
```

#### Quick Training (for testing)
```bash
# Reduced epochs for quick testing
python train.py --model resnet --epochs 10
python train.py --model plain --epochs 10
```

### 3. Compare Results

```bash
# Generate comparison plots and analysis
python src/evaluation/compare_models.py --results_dir ./results --output_dir ./comparison_plots
```

### 4. Interactive Analysis

```bash
# Launch Jupyter notebook for detailed analysis
jupyter notebook notebooks/ResNet_CIFAR10_Comparison.ipynb
```

## Detailed Usage

### Command Line Arguments

#### Training Script (`train.py`)

```bash
python train.py [OPTIONS]

Options:
  --model {resnet,plain}     Model type to train (required)
  --epochs INT              Number of training epochs [default: 200]
  --batch_size INT          Batch size for training [default: 128]
  --lr FLOAT                Learning rate [default: 0.1]
  --weight_decay FLOAT      Weight decay [default: 1e-4]
  --seed INT                Random seed [default: 42]
  --data_dir PATH           Data directory [default: ./data]
  --save_dir PATH           Save directory for results [default: ./results]
  --device {auto,cuda,cpu}  Device to use [default: auto]
```

#### Comparison Script (`compare_models.py`)

```bash
python src/evaluation/compare_models.py [OPTIONS]

Options:
  --results_dir PATH        Directory containing training results [default: ./results]
  --output_dir PATH         Directory to save comparison plots [default: ./comparison_plots]
```

### Example Workflows

#### Full Experiment (Recommended)
```bash
# 1. Train both models with full epochs
python train.py --model resnet --epochs 200 --save_dir ./results/full_experiment
python train.py --model plain --epochs 200 --save_dir ./results/full_experiment

# 2. Generate comprehensive comparison
python src/evaluation/compare_models.py --results_dir ./results/full_experiment

# 3. Analyze in Jupyter notebook
jupyter notebook notebooks/ResNet_CIFAR10_Comparison.ipynb
```

#### Quick Experiment (for testing)
```bash
# 1. Quick training with reduced epochs
python train.py --model resnet --epochs 20 --save_dir ./results/quick_test
python train.py --model plain --epochs 20 --save_dir ./results/quick_test

# 2. Quick comparison
python src/evaluation/compare_models.py --results_dir ./results/quick_test
```

#### GPU Training
```bash
# Explicitly use GPU (if available)
python train.py --model resnet --device cuda --batch_size 256
python train.py --model plain --device cuda --batch_size 256
```

#### CPU Training
```bash
# Force CPU training (slower but works without GPU)
python train.py --model resnet --device cpu --batch_size 64
python train.py --model plain --device cpu --batch_size 64
```

## Understanding the Output

### Training Output
During training, you'll see:
```
Epoch 1/200 (45.2s)
  Train Loss: 1.8234, Train Acc: 32.45%
  Test Loss: 1.6789, Test Acc: 38.21%
  Gradient Norm: 0.8234
```

### Results Structure
```
results/
├── resnet_20251007_143022/
│   ├── checkpoints/
│   │   └── best_model.pth
│   ├── logs/
│   │   └── tensorboard_logs/
│   └── results.pkl
└── plain_20251007_144530/
    ├── checkpoints/
    ├── logs/
    └── results.pkl
```

### Generated Plots
After comparison, you'll get:
```
comparison_plots/
├── training_curves.png      # Loss and accuracy curves
├── model_comparison.png     # Comprehensive comparison
└── gradient_flow.png        # Gradient analysis
```

## Customization

### Modifying Hyperparameters

Edit the training script or use command line arguments:
```bash
# Custom learning rate schedule
python train.py --model resnet --lr 0.01 --weight_decay 5e-4

# Different batch size
python train.py --model resnet --batch_size 64
```

### Adding New Models

1. Create new model in `src/models/`
2. Add to `__init__.py`
3. Modify training script to include new model

### Custom Data Augmentation

Edit `src/data/dataset.py` to modify data preprocessing:
```python
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add this
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
```

## Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --model resnet --batch_size 64

# Or use CPU
python train.py --model resnet --device cpu
```

#### Slow Training
```bash
# Reduce epochs for testing
python train.py --model resnet --epochs 50

# Use smaller model (modify code to use fewer layers)
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install -r requirements.txt

# Or install individually
pip install torch torchvision matplotlib seaborn pandas scikit-learn tqdm tensorboard
```

### Performance Tips

1. **Use GPU**: Significantly faster training
2. **Increase batch size**: Better GPU utilization (if memory allows)
3. **Use multiple workers**: Set `num_workers=4` in data loader
4. **Mixed precision**: Add `--amp` flag (if implemented)

### Monitoring Training

#### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir results/

# Open browser to http://localhost:6006
```

#### Real-time Monitoring
```bash
# Watch training progress
tail -f results/resnet_*/logs/training.log
```

## Expected Results

### Performance Benchmarks
- **ResNet-18**: ~92-94% test accuracy
- **Plain CNN-18**: ~87-89% test accuracy
- **Training time**: 2-3 hours on GPU, 8-12 hours on CPU

### Key Observations
1. ResNet converges faster and more stably
2. Plain CNN shows more training oscillations
3. ResNet maintains better gradient flow
4. Performance gap increases with network depth

## Next Steps

After running the basic comparison:

1. **Experiment with deeper networks**: Try ResNet-34 vs Plain CNN-34
2. **Different datasets**: Adapt code for CIFAR-100 or ImageNet
3. **Architecture variations**: Implement ResNeXt, DenseNet
4. **Optimization techniques**: Try different optimizers, learning rates
5. **Analysis extensions**: Add more visualization and analysis tools

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the Jupyter notebook for detailed examples
3. Examine the research report for theoretical background
4. Open an issue on the GitHub repository