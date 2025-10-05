"""
Training script for transformer-based models on PPMI dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import DataPreprocessor
from models.transformer_models import TransformerModels, TabularDataset


def main():
    """Main training function."""
    print("Starting Transformer Models Training...")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize preprocessor and load data
    preprocessor = DataPreprocessor()
    
    # Use all three datasets
    file_paths = [
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20241211.csv",
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20250321.csv",
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20250714.csv"
    ]
    
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
    
    # Create datasets and data loaders
    train_dataset = TabularDataset(X_train, y_train)
    test_dataset = TabularDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Split training data for validation
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader_split = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    # Initialize transformer models
    transformer_trainer = TransformerModels()
    
    # Model configurations
    models_config = {
        'transformer_small': {
            'type': 'transformer',
            'params': {
                'd_model': 64,
                'nhead': 4,
                'num_layers': 2,
                'dropout': 0.1
            }
        },
        'transformer_medium': {
            'type': 'transformer',
            'params': {
                'd_model': 128,
                'nhead': 8,
                'num_layers': 3,
                'dropout': 0.2
            }
        },
        'transformer_large': {
            'type': 'transformer',
            'params': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.3
            }
        },
        'feedforward': {
            'type': 'feedforward',
            'params': {
                'hidden_dims': [256, 128, 64],
                'dropout': 0.3
            }
        }
    }
    
    results = {}
    
    # Train each model
    for model_name, config in models_config.items():
        print(f"\n{'='*20} Training {model_name} {'='*20}")
        
        # Create model
        model = transformer_trainer.create_model(
            config['type'], 
            X_train.shape[1], 
            len(np.unique(y_train)),
            **config['params']
        )
        
        print(f"Model architecture: {model}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        history = transformer_trainer.train_model(
            model, 
            train_loader_split, 
            val_loader,
            epochs=100,
            lr=0.001,
            weight_decay=1e-4,
            class_weights=class_weights_tensor
        )
        
        # Evaluate on test set
        test_results = transformer_trainer.evaluate_model(model, test_loader)
        
        # Store results
        results[model_name] = {
            'model': model,
            'history': history,
            'test_results': test_results
        }
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Classification Report:\n{test_results['classification_report']}")
        
        # Save model
        transformer_trainer.save_model(model, model_name)
        
        # Plot training history
        transformer_trainer.plot_training_history(history, model_name)
        
        # Plot confusion matrix
        transformer_trainer.plot_confusion_matrix(
            test_results['confusion_matrix'], 
            model_name,
            class_names=['HC', 'PD', 'SWEDD', 'PRODROMAL']
        )
    
    # Cross-validation for best model
    print(f"\n{'='*20} Cross-Validation {'='*20}")
    
    # Find best model based on test accuracy
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_results']['accuracy'])
    best_config = models_config[best_model_name]
    
    print(f"Best model: {best_model_name}")
    print("Performing 5-fold cross-validation...")
    
    cv_scores = transformer_trainer.cross_validate(
        best_config['type'],
        X_train,
        y_train,
        cv_folds=5,
        **best_config['params']
    )
    
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    
    # Summary comparison
    print(f"\n{'='*20} Model Comparison {'='*20}")
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Test Accuracy': f"{result['test_results']['accuracy']:.4f}",
            'Parameters': f"{sum(p.numel() for p in result['model'].parameters()):,}"
        })
    
    # Create comparison plot
    model_names = list(results.keys())
    accuracies = [results[name]['test_results']['accuracy'] for name in model_names]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Comparison - Test Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('notebooks/transformer_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print comparison table
    print("\nModel Performance Summary:")
    print("-" * 60)
    for data in comparison_data:
        print(f"{data['Model']:<20} | Accuracy: {data['Test Accuracy']:<8} | Params: {data['Parameters']}")
    
    print(f"\nTraining completed! Models saved in 'models/saved/' directory.")
    print(f"Visualizations saved in 'notebooks/' directory.")


if __name__ == "__main__":
    main()