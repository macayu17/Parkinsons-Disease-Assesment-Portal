"""
Transformer-based models for Parkinson's disease classification.
This module implements various transformer architectures adapted for tabular data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import joblib
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns


class TabularDataset(Dataset):
    """Custom dataset for tabular data."""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TabularTransformer(nn.Module):
    """Transformer model adapted for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super(TabularTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Feature embedding - each feature gets its own embedding
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])
        
        # Positional encoding for features
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Create embeddings for each feature
        feature_embeddings = []
        for i in range(self.input_dim):
            # Extract feature i and embed it
            feature_val = x[:, i:i+1]  # (batch_size, 1)
            embedded = self.feature_embeddings[i](feature_val)  # (batch_size, d_model)
            feature_embeddings.append(embedded)
        
        # Stack embeddings: (batch_size, input_dim, d_model)
        x = torch.stack(feature_embeddings, dim=1)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Global average pooling across features
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Classification
        output = self.classifier(x)
        
        return output


class FeedForwardNetwork(nn.Module):
    """Simple feed-forward network for comparison."""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = [256, 128, 64], 
                 dropout: float = 0.3):
        super(FeedForwardNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class TransformerModels:
    """Class to handle training and evaluation of transformer models."""
    
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.training_history = {}
        
    def create_model(self, model_type: str, input_dim: int, num_classes: int, **kwargs):
        """Create a model of specified type."""
        if model_type == 'transformer':
            model = TabularTransformer(input_dim, num_classes, **kwargs)
        elif model_type == 'feedforward':
            model = FeedForwardNetwork(input_dim, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, epochs: int = 100, 
                   lr: float = 0.001, weight_decay: float = 1e-4, 
                   class_weights: torch.Tensor = None):
        """Train a model."""
        
        # Loss function with class weights
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 20
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            scheduler.step(val_loss)
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies
        }
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model on test set."""
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())
        
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(all_targets, all_predictions)
        cm = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'targets': all_targets,
            'classification_report': report,
            'confusion_matrix': cm
        }
    
    def cross_validate(self, model_type: str, X, y, cv_folds: int = 5, **model_kwargs):
        """Perform cross-validation."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nFold {fold + 1}/{cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create datasets and loaders
            train_dataset = TabularDataset(X_train_fold, y_train_fold)
            val_dataset = TabularDataset(X_val_fold, y_val_fold)
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
            
            # Create and train model
            model = self.create_model(model_type, X.shape[1], len(np.unique(y)), **model_kwargs)
            
            # Calculate class weights for this fold
            class_counts = np.bincount(y_train_fold)
            class_weights = torch.FloatTensor(len(class_counts) / (len(class_counts) * class_counts))
            
            # Train model
            history = self.train_model(model, train_loader, val_loader, 
                                     epochs=50, class_weights=class_weights)
            
            # Evaluate on validation set
            results = self.evaluate_model(model, val_loader)
            cv_scores.append(results['accuracy'])
            
            print(f"Fold {fold + 1} Accuracy: {results['accuracy']:.4f}")
        
        return cv_scores
    
    def save_model(self, model, model_name: str, save_dir: str = "models/saved"):
        """Save trained model."""
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}_transformer.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_type: str, model_name: str, input_dim: int, 
                   num_classes: int, save_dir: str = "models/saved", **model_kwargs):
        """Load trained model."""
        model_path = os.path.join(save_dir, f"{model_name}_transformer.pth")
        
        model = self.create_model(model_type, input_dim, num_classes, **model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        return model
    
    def plot_training_history(self, history: Dict, model_name: str, save_dir: str = "notebooks"):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(history['train_losses'], label='Train Loss')
        ax1.plot(history['val_losses'], label='Validation Loss')
        ax1.set_title(f'{model_name} - Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['val_accuracies'], label='Validation Accuracy', color='green')
        ax2.set_title(f'{model_name} - Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name: str, class_names: list = None, 
                            save_dir: str = "notebooks"):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_transformer_confusion_matrix.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()