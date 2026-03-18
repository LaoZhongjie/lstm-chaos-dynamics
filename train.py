"""
Training script for RNN model (LSTM/GRU/simple RNN) with checkpointing for chaos analysis
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

import config
from model import RNN
from config_saver import save_experiment_config
from data_loader import IMDBDataLoader
from seed_utils import HierarchicalSeedManager

class LSTMTrainer:
    """Trainer class for RNN model with comprehensive logging"""
    
    def __init__(self, seed_manager=None):
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Training RNN model (cell_type={config.RNN_CELL_TYPE})")
        self.seed_manager = seed_manager or HierarchicalSeedManager(config.RANDOM_SEED)
        
        # Create directories
        self.create_directories()
        
        # Initialize data loader
        self.data_loader = IMDBDataLoader(seed_manager=self.seed_manager)
        
        # Initialize model, optimizer, and loss function
        self.model = None
        self.optimizer = None
        # Model returns raw logits (no sigmoid), so use numerically-stable BCE-with-logits.
        self.criterion = nn.BCEWithLogitsLoss()
        self.embedding_fix = config.EMBEDDING_FIX
        self.fc_fix = config.FC_FIX
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': [],
            'epochs': [],
            'grad_norms': []
        }

    def load_weights_from_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return None, None
        
        print(f"Loading weights from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        embedding_weights = None
        fc_weights = None
        
        if self.embedding_fix:
            embedding_weights = state_dict['embedding.weight'].cpu().numpy()
            print(f"[init model]Extracted embedding weights: shape {embedding_weights.shape}")
            
        if self.fc_fix:
            fc_weight = state_dict['fc.weight'].cpu().numpy()
            fc_bias = state_dict['fc.bias'].cpu().numpy()
            fc_weights = (fc_weight, fc_bias)
            print(f"[init model]Extracted fc weights: weight shape {fc_weight.shape}, bias shape {fc_bias.shape}")
        
        return embedding_weights, fc_weights

    def initialize_model(self, vocab_size, pretrained_checkpoint=config.PRETRAINED_CHECKPOINT):
        
        if self.embedding_fix or self.fc_fix:
            embedding_weights, fc_weights = self.load_weights_from_checkpoint(pretrained_checkpoint)
        else:
            embedding_weights = None
            fc_weights = None
        
        self.model = RNN(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_size=config.HIDDEN_SIZE,
            num_classes=config.NUM_CLASSES,
            embedding_weights=embedding_weights,
            fc_weights=fc_weights,
            seed_manager=self.seed_manager,
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
        )
        
        print(f"RNN parameters: {sum(p.numel() for p in self.model.parameters())}")
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (data, target, lengths) in enumerate(progress_bar):
            data = data.to(self.device)
            target = target.to(self.device)
            lengths = lengths.to(self.device)
            
            self.optimizer.zero_grad()
            logits, _ = self.model(data, lengths)
            logits = logits.squeeze(-1)
            
            loss = self.criterion(logits, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            predicted = (probs > 0.5).float()
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%',
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target, lengths) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                lengths = lengths.to(self.device)
                
                logits, _ = self.model(data, lengths)
                logits = logits.squeeze(-1)
                
                loss = self.criterion(logits, target)
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                predicted = (probs > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, train_loss, test_loss, train_acc, test_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'vocab_size': self.data_loader.vocab_size,
        }
        
        checkpoint_path = os.path.join(config.CHECKPOINT_PATH, f'model_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if not hasattr(self, 'best_test_loss') or test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_epoch = epoch
            best_path = os.path.join(config.CHECKPOINT_PATH, 'best_model.pt')
            torch.save(checkpoint, best_path)
    
    def save_training_history(self):
        """Save training history to JSON file"""
        history_path = os.path.join(config.RESULTS_PATH, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def train(self, max_epochs=None):
        """Main training loop"""
        if max_epochs is None:
            max_epochs = config.MAX_EPOCHS
            
        print(f"Starting training for {max_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(1, max_epochs + 1):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluation
            test_loss, test_acc = self.evaluate(epoch)
            
            # Save to history
            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['test_loss'].append(test_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['test_accuracy'].append(test_acc)
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, test_loss, train_acc, test_acc)
            
            # Print epoch summary
            print(f'Epoch {epoch:4d}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
            
            # Save history periodically
            if epoch % 10 == 0:
                self.save_training_history()
        
        print("=" * 60)
        print(f"Training completed! Best epoch: {self.best_epoch}, Best test loss: {self.best_test_loss:.4f}")
        
        # Final save
        self.save_training_history()
        save_experiment_config(extra={
            "run_type": "training",
            "best_epoch": self.best_epoch,
            "best_test_loss": float(self.best_test_loss),
            "total_epochs": max_epochs,
            "vocab_size": self.data_loader.vocab_size,
            "pretrained_checkpoint": config.PRETRAINED_CHECKPOINT,
            "embedding_fix": config.EMBEDDING_FIX,
            "fc_fix": config.FC_FIX,
        })
        
        return self.training_history
    
    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(config.RESULTS_PATH, exist_ok=True)
        os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)

    def load_data(self):
        print("Loading IMDB dataset...")
        X_train, X_test, y_train, y_test = self.data_loader.load_data()
        
        self.train_loader, self.test_loader, self.test_dataset = \
            self.data_loader.create_data_loaders(X_train, X_test, y_train, y_test)
        
        return self.data_loader.vocab_size
