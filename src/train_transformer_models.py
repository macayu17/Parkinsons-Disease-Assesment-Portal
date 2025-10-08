"""
Training script for transformer-based models (DistilBERT, BioBERT, PubMedBERT) on PPMI dataset
with RAG integration.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_preprocessing import DataPreprocessor
from models.transformer_models import (
    TabularDataset, 
    DistilBERTForTabular, 
    BioBERTForTabular, 
    PubMedBERTForTabular
)
from document_manager import DocumentManager


def main():
    """Main training function."""
    print("Starting Transformer Models Training with RAG Integration...")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor()
    
    # Use all four datasets
    file_paths = [
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20241211.csv",
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20250321.csv",
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20250714.csv",
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20251105.csv"
    ]
    
    # Configure GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading and preprocessing data from multiple datasets...")
    # Load and combine all datasets
    df = preprocessor.load_multiple_datasets(file_paths)
    
    # Process the combined dataset
    if df is not None:
        print(f"Combined dataset shape: {df.shape}")
        
        # Preprocess data
        df = preprocessor.remove_leakage_features(df)
        
        # Prepare data using existing method
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    if X_train is None:
        print("Error: Failed to load data")
        return
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Convert DataFrames to numpy arrays if needed
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights_tensor = torch.FloatTensor(class_weights)
    
    print(f"Class weights: {class_weights}")
    
    # Get feature names for better text representation
    feature_names = preprocessor.get_feature_names()
    
    # Create datasets and data loaders with feature names
    train_dataset = TabularDataset(X_train, y_train, feature_names)
    test_dataset = TabularDataset(X_test, y_test, feature_names)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Split training data for validation
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader_split = DataLoader(train_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    # Initialize document manager for RAG
    doc_manager = DocumentManager(docs_dir="../docs")
    print(f"Loaded {doc_manager.get_document_count()['total']} documents for RAG")
    
    # Function to get RAG context for a sample
    def get_rag_context(sample_features, feature_names):
        # Convert sample features to a descriptive text
        feature_desc = {name: float(val) for name, val in zip(feature_names, sample_features)}
        
        # Create a query from key features
        query_parts = []
        
        # Add key symptoms with their values
        symptoms = {
            'tremor': feature_desc.get('sym_tremor', 0),
            'rigidity': feature_desc.get('sym_rigid', 0),
            'bradykinesia': feature_desc.get('sym_brady', 0),
            'postural instability': feature_desc.get('sym_posins', 0)
        }
        
        for symptom, severity in symptoms.items():
            if severity > 0:
                query_parts.append(f"{symptom} severity:{severity}")
        
        # Add cognitive factors
        if 'moca' in feature_desc:
            moca = feature_desc.get('moca', 30)
            if moca < 26:
                query_parts.append("cognitive impairment")
        
        # Add demographic factors
        if 'age' in feature_desc:
            age = feature_desc.get('age', 0)
            if age > 0:
                query_parts.append(f"age {int(age)}")
                
        # Add family history if present
        if feature_desc.get('fampd', 0) > 0:
            query_parts.append("family history Parkinson's disease")
        
        # Create query
        query = "Parkinson's disease " + " ".join(query_parts)
        
        # Get relevant passages
        passages = doc_manager.extract_relevant_passages(query, top_k=2)
        
        if not passages:
            return ""
        
        # Format context
        context = ""
        for passage in passages:
            context += f"From '{passage['doc_title']}': {passage['text'][:300]}... "
        
        return context
    
    # Initialize the new transformer models
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Improved hyperparameters for better accuracy
    dropout_rate = 0.2  # Increased dropout for better generalization
    
    # Create models with optimized hyperparameters
    distilbert_model = DistilBERTForTabular(input_dim, num_classes, dropout=dropout_rate).to(device)
    biobert_model = BioBERTForTabular(input_dim, num_classes, dropout=dropout_rate).to(device)
    pubmedbert_model = PubMedBERTForTabular(input_dim, num_classes, dropout=dropout_rate).to(device)
    
    # Define optimizers with improved learning rates
    distilbert_optimizer = torch.optim.AdamW(distilbert_model.parameters(), lr=2e-4, weight_decay=0.01)
    biobert_optimizer = torch.optim.AdamW(biobert_model.parameters(), lr=2e-4, weight_decay=0.01)
    pubmedbert_optimizer = torch.optim.AdamW(pubmedbert_model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Define loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    
    # Learning rate schedulers for better convergence
    distilbert_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(distilbert_optimizer, mode='min', factor=0.5, patience=3)
    biobert_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(biobert_optimizer, mode='min', factor=0.5, patience=3)
    pubmedbert_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(pubmedbert_optimizer, mode='min', factor=0.5, patience=3)
    
    # Training parameters
    num_epochs = 30  # Increased epochs for better convergence
    patience = 5  # Early stopping patience
    models = {
        "distilbert": (distilbert_model, distilbert_optimizer, distilbert_scheduler),
        "biobert": (biobert_model, biobert_optimizer, biobert_scheduler),
        "pubmedbert": (pubmedbert_model, pubmedbert_optimizer, pubmedbert_scheduler)
    }
    
    # Training loop for all models
    print("Starting training for all models...")
    results = {}
    
    for model_name, (model, optimizer, scheduler) in models.items():
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        
        print(f"Model architecture: {model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training variables
        best_val_loss = float('inf')
        best_model_state = None
        history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}
        early_stop_counter = 0
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_idx, (data, targets) in enumerate(train_loader_split):
                data, targets = data.to(device), targets.to(device)
                
                # Get RAG context for each sample in batch
                batch_contexts = []
                for i in range(data.shape[0]):
                    context = get_rag_context(data[i].cpu().numpy(), feature_names)
                    batch_contexts.append(context)
                
                # Forward pass with RAG context
                outputs = model(data, batch_contexts)
                loss = criterion(outputs, targets)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader_split)}, Loss: {loss.item():.4f}")
            
            avg_train_loss = train_loss / len(train_loader_split)
            history['train_loss'].append(avg_train_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    
                    # Get RAG context for each sample in batch
                    batch_contexts = []
                    for i in range(data.shape[0]):
                        context = get_rag_context(data[i].cpu().numpy(), feature_names)
                        batch_contexts.append(context)
                    
                    outputs = model(data, batch_contexts)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
                    
                    # Store predictions and targets for metrics
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_loader)
            accuracy = 100 * correct / total
            
            # Calculate additional metrics
            val_f1 = f1_score(all_targets, all_preds, average='weighted')
            val_precision = precision_score(all_targets, all_preds, average='weighted')
            val_recall = recall_score(all_targets, all_preds, average='weighted')
            
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(accuracy)
            
            # Update learning rate scheduler
            scheduler.step(avg_val_loss)
            
            print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Accuracy: {accuracy:.2f}%, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0
                print(f"New best model for {model_name}!")
            else:
                early_stop_counter += 1
                
            # Early stopping
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save the best model
        torch.save(best_model_state, f"../models/{model_name}_model.pth")
        print(f"Best {model_name} model saved with validation loss: {best_val_loss:.4f}")
        
        # Evaluate on test set
        model.load_state_dict(best_model_state)
        model.eval()
        
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(device), targets.to(device)
                
                # Get RAG context for each sample in batch
                batch_contexts = []
                for i in range(data.shape[0]):
                    context = get_rag_context(data[i].cpu().numpy(), feature_names)
                    batch_contexts.append(context)
                
                outputs = model(data, batch_contexts)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_accuracy = np.mean(np.array(all_preds) == np.array(all_targets))
        test_f1 = f1_score(all_targets, all_preds, average='weighted')
        test_precision = precision_score(all_targets, all_preds, average='weighted')
        test_recall = recall_score(all_targets, all_preds, average='weighted')
        classification_rep = classification_report(all_targets, all_preds, target_names=['HC', 'PD', 'SWEDD', 'PRODROMAL'])
        conf_matrix = confusion_matrix(all_targets, all_preds)
        
        # Store results
        results[model_name] = {
            'model': model,
            'history': history,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'targets': all_targets
        }
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['HC', 'PD', 'SWEDD', 'PRODROMAL'],
                   yticklabels=['HC', 'PD', 'SWEDD', 'PRODROMAL'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'{model_name} - Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f'../notebooks/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print comprehensive results
        print(f"\n{model_name} Results:")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 Score: {test_f1:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Classification Report:\n{classification_rep}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        
        # Plot enhanced training history with learning rate
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss', color='#3498db')
        plt.plot(history['val_loss'], label='Validation Loss', color='#e74c3c')
        plt.title(f'{model_name} - Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(history['val_acc'], label='Validation Accuracy', color='#2ecc71')
        plt.title(f'{model_name} - Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(history['lr'], label='Learning Rate', color='#9b59b6')
        plt.title(f'{model_name} - Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add class distribution pie chart
        plt.subplot(2, 2, 4)
        unique, counts = np.unique(all_targets, return_counts=True)
        class_names = ['HC', 'PD', 'SWEDD', 'PRODROMAL']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        plt.pie(counts, labels=[f"{class_names[i]} ({counts[i]})" for i in range(len(unique))], 
                autopct='%1.1f%%', startangle=90, colors=colors)
        plt.axis('equal')
        plt.title(f'{model_name} - Test Set Class Distribution')
        
        plt.tight_layout()
        plt.savefig(f'../notebooks/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Summary comparison
    print(f"\n{'='*20} Model Comparison {'='*20}")
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Test Accuracy': f"{result['test_accuracy']:.4f}",
            'F1 Score': f"{result['test_f1']:.4f}",
            'Precision': f"{result['test_precision']:.4f}",
            'Recall': f"{result['test_recall']:.4f}",
            'Parameters': f"{sum(p.numel() for p in result['model'].parameters()):,}"
        })
    
    # Create comprehensive comparison plots
    model_names = list(results.keys())
    metrics = {
        'Accuracy': [results[name]['test_accuracy'] for name in model_names],
        'F1 Score': [results[name]['test_f1'] for name in model_names],
        'Precision': [results[name]['test_precision'] for name in model_names],
        'Recall': [results[name]['test_recall'] for name in model_names]
    }
    
    # Plot all metrics in a single figure
    plt.figure(figsize=(15, 10))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        plt.subplot(2, 2, i+1)
        bars = plt.bar(model_names, metric_values, color=colors)
        plt.title(f'Model Comparison - {metric_name}')
        plt.xlabel('Model')
        plt.ylabel(metric_name)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('../notebooks/transformer_models_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a radar chart for comprehensive model comparison
    plt.figure(figsize=(10, 8))
    
    # Set up the radar chart
    categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Set up subplot
    ax = plt.subplot(111, polar=True)
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories)
    
    # Draw the model performance on the chart
    for i, model_name in enumerate(model_names):
        values = [
            results[model_name]['test_accuracy'],
            results[model_name]['test_f1'],
            results[model_name]['test_precision'],
            results[model_name]['test_recall']
        ]
        values += values[:1]  # Close the loop
        
        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Comparison')
    plt.tight_layout()
    plt.savefig('../notebooks/transformer_models_radar_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comprehensive comparison table
    print("\nModel Performance Summary:")
    print("-" * 100)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10} | {'Parameters':<15}")
    print("-" * 100)
    for data in comparison_data:
        print(f"{data['Model']:<15} | {data['Test Accuracy']:<10} | {data['F1 Score']:<10} | {data['Precision']:<10} | {data['Recall']:<10} | {data['Parameters']:<15}")
    
    # Find best model based on F1 score (more comprehensive than accuracy alone)
    best_model_name_acc = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
    best_model_name_f1 = max(results.keys(), key=lambda k: results[k]['test_f1'])
    
    print(f"\nBest model by accuracy: {best_model_name_acc} with accuracy: {results[best_model_name_acc]['test_accuracy']:.4f}")
    print(f"Best model by F1 score: {best_model_name_f1} with F1 score: {results[best_model_name_f1]['test_f1']:.4f}")
    
    # Save detailed metrics to CSV for further analysis
    metrics_df = pd.DataFrame(comparison_data)
    metrics_df.to_csv('../notebooks/transformer_models_metrics.csv', index=False)
    
    print(f"\nTraining completed! Models saved in '../models/' directory.")
    print(f"Visualizations saved in '../notebooks/' directory.")


if __name__ == "__main__":
    main()