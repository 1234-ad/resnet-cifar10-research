"""
Training Logic for ResNet vs Plain CNN Comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
from tqdm import tqdm
import numpy as np


class Trainer:
    """Training class for model comparison"""
    
    def __init__(self, model, train_loader, test_loader, device='cuda', 
                 lr=0.1, weight_decay=1e-4, log_dir='./logs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr, 
                                  momentum=0.9, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[60, 120, 160], gamma=0.2
        )
        
        # Logging
        self.writer = SummaryWriter(log_dir)
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.gradient_norms = []
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        gradient_norms = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Calculate gradient norm for analysis
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            gradient_norms.append(total_norm)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        avg_grad_norm = np.mean(gradient_norms)
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        self.gradient_norms.append(avg_grad_norm)
        
        # Log to tensorboard
        self.writer.add_scalar('Train/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
        self.writer.add_scalar('Train/GradientNorm', avg_grad_norm, epoch)
        self.writer.add_scalar('Train/LearningRate', 
                              self.optimizer.param_groups[0]['lr'], epoch)
        
        return epoch_loss, epoch_acc
    
    def test_epoch(self, epoch):
        """Test for one epoch"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = test_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        
        self.test_losses.append(epoch_loss)
        self.test_accuracies.append(epoch_acc)
        
        # Log to tensorboard
        self.writer.add_scalar('Test/Loss', epoch_loss, epoch)
        self.writer.add_scalar('Test/Accuracy', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=200, save_path='./checkpoints'):
        """Full training loop"""
        os.makedirs(save_path, exist_ok=True)
        best_acc = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train and test
            train_loss, train_acc = self.train_epoch(epoch)
            test_loss, test_acc = self.test_epoch(epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            print(f'  Gradient Norm: {self.gradient_norms[-1]:.4f}')
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                    'train_losses': self.train_losses,
                    'train_accuracies': self.train_accuracies,
                    'test_losses': self.test_losses,
                    'test_accuracies': self.test_accuracies,
                    'gradient_norms': self.gradient_norms,
                }, os.path.join(save_path, 'best_model.pth'))
        
        self.writer.close()
        print(f'Training completed! Best test accuracy: {best_acc:.2f}%')
        return best_acc
    
    def get_training_history(self):
        """Return training history for analysis"""
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
            'gradient_norms': self.gradient_norms
        }