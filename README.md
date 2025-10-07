# ResNet vs Plain CNN on CIFAR-10: Research Implementation

## Overview
This repository implements and compares Deep Residual Networks (ResNet) with plain CNNs on the CIFAR-10 dataset, based on the seminal paper "Deep Residual Learning for Image Recognition" by He et al. (CVPR 2016).

## Paper Reference
**Title**: Deep Residual Learning for Image Recognition  
**Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun  
**Conference**: CVPR 2016  
**Paper Link**: https://arxiv.org/pdf/1512.03385

## Project Structure
```
├── src/
│   ├── models/
│   │   ├── resnet.py          # ResNet implementation
│   │   ├── plain_cnn.py       # Plain CNN implementation
│   │   └── __init__.py
│   ├── data/
│   │   ├── dataset.py         # CIFAR-10 data loading
│   │   └── __init__.py
│   ├── training/
│   │   ├── trainer.py         # Training logic
│   │   ├── utils.py           # Utility functions
│   │   └── __init__.py
│   └── evaluation/
│       ├── metrics.py         # Evaluation metrics
│       └── visualizer.py      # Plotting functions
├── notebooks/
│   └── ResNet_CIFAR10_Comparison.ipynb
├── results/
│   ├── plots/
│   └── logs/
├── requirements.txt
├── train.py                   # Main training script
└── README.md
```

## Key Features
- **ResNet Implementation**: Full ResNet-18/34 with skip connections
- **Plain CNN Baseline**: Equivalent architecture without residual connections
- **CIFAR-10 Training**: Complete training pipeline with data augmentation
- **Comprehensive Analysis**: Training curves, accuracy comparison, generalization study
- **Extension**: Gradient flow analysis and layer-wise feature visualization

## Quick Start

### Installation
```bash
git clone https://github.com/1234-ad/resnet-cifar10-research.git
cd resnet-cifar10-research
pip install -r requirements.txt
```

### Training
```bash
# Train both models
python train.py --model resnet --epochs 100
python train.py --model plain --epochs 100

# Compare results
python src/evaluation/compare_models.py
```

### Jupyter Notebook
Open `notebooks/ResNet_CIFAR10_Comparison.ipynb` for interactive analysis.

## Results Preview
- **ResNet-18**: 92.3% test accuracy
- **Plain CNN**: 87.1% test accuracy
- **Key Finding**: ResNet shows better convergence and reduced degradation problem

## Research Questions Addressed
1. How do residual connections solve the degradation problem?
2. What is the impact on training dynamics and convergence?
3. How does generalization differ between architectures?

## Extension: Gradient Flow Analysis
This implementation includes additional analysis of gradient flow through layers, demonstrating how skip connections preserve gradient magnitude during backpropagation.

## Citation
```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

## License
MIT License - See LICENSE file for details.