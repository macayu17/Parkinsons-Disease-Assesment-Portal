from data_preprocessing import DataPreprocessor
from models.traditional_ml import TraditionalMLModels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

def plot_confusion_matrices(results, save_dir='../notebooks'):
    """Plot confusion matrices for all models."""
    os.makedirs(save_dir, exist_ok=True)
    
    for model_name, result in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
        plt.close()

def plot_roc_curves(results, y_test, save_dir='../notebooks'):
    """Plot ROC curves for all models using one-vs-rest approach."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Get number of classes
    n_classes = len(np.unique(y_test))
    
    # Binarize the labels for one-vs-rest ROC curves
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    # Create subplots for each class
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    # Plot ROC curves for each class
    for i in range(n_classes):
        for model_name, result in results.items():
            # Get probabilities for the current class
            y_prob = result['probabilities'][:, i]
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob)
            roc_auc = auc(fpr, tpr)
            
            axes[i].plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
            axes[i].plot([0, 1], [0, 1], 'k--')
            axes[i].set_xlim([0.0, 1.0])
            axes[i].set_ylim([0.0, 1.05])
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'ROC Curve - Class {i}')
            axes[i].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()

def main():
    # Initialize preprocessor and prepare data
    preprocessor = DataPreprocessor()
    
    # Use all three datasets
    file_paths = [
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20241211.csv",
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20250321.csv",
        "D:/5th Semester/Projects/MiniProject/Try1/PPMI_Curated_Data_Cut_Public_20250714.csv"
    ]
    
    # Load and combine all datasets
    df = preprocessor.load_multiple_datasets(file_paths)
    
    # Process the combined dataset
    if df is not None:
        print(f"Combined dataset shape: {df.shape}")
        
        # Preprocess data
        df = preprocessor.remove_leakage_features(df)
        
        # Prepare data using existing method
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print("\nClass weights:", class_weight_dict)
    
    # Initialize models with class weights
    ml_models = TraditionalMLModels()
    
    # Update model parameters to handle class imbalance
    ml_models.models['lightgbm'] = LGBMClassifier(
        random_state=42,
        class_weight=class_weight_dict,
        objective='multiclass',
        num_class=len(class_weight_dict)
    )
    ml_models.models['xgboost'] = XGBClassifier(
        random_state=42,
        objective='multi:softmax',
        num_class=len(class_weight_dict)
    )
    ml_models.models['svm'] = SVC(
        random_state=42,
        class_weight=class_weight_dict,
        probability=True,
        decision_function_shape='ovr'
    )
    
    print("\nTraining models...")
    cv_scores = ml_models.train_all_models(X_train, y_train)
    
    # Print cross-validation scores
    print("\nCross-validation scores:")
    for model_name, score in cv_scores.items():
        print(f"{model_name}: {score:.4f}")
    
    # Evaluate models on test set
    print("\nEvaluating models on test set...")
    results = ml_models.evaluate_all_models(X_test, y_test)
    
    # Print test set results
    print("\nTest set results:")
    for model_name, result in results.items():
        print(f"\n{model_name.upper()} Results:")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print("\nClassification Report:")
        print(result['classification_report'])
    
    # Plot confusion matrices and ROC curves
    plot_confusion_matrices(results)
    plot_roc_curves(results, y_test)
    
    # Save models
    print("\nSaving models...")
    ml_models.save_models()

if __name__ == "__main__":
    main()