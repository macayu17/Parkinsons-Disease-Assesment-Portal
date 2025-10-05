import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import joblib
import os

class TraditionalMLModels:
    def __init__(self, save_dir='../models/saved'):
        """Initialize the traditional ML models class.
        
        Args:
            save_dir (str): Directory to save trained models
        """
        self.models = {
            'lightgbm': None,
            'xgboost': None,
            'svm': None
        }
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model with hyperparameter tuning."""
        if self.models['lightgbm'] is None:
            self.models['lightgbm'] = LGBMClassifier(random_state=42)
            
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'num_leaves': [31, 63],
            'objective': ['multiclass'],
            'metric': ['multi_logloss'],
            'num_class': [4]  # Number of unique classes in target
        }
        
        grid_search = GridSearchCV(self.models['lightgbm'], param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['lightgbm'] = grid_search.best_estimator_
        print("LightGBM best parameters:", grid_search.best_params_)
        return grid_search.best_score_
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model with hyperparameter tuning."""
        if self.models['xgboost'] is None:
            self.models['xgboost'] = XGBClassifier(random_state=42)
            
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3],
            'objective': ['multi:softmax'],
            'num_class': [4],  # Number of unique classes in target
            'eval_metric': ['mlogloss']
        }
        
        grid_search = GridSearchCV(self.models['xgboost'], param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['xgboost'] = grid_search.best_estimator_
        print("XGBoost best parameters:", grid_search.best_params_)
        return grid_search.best_score_
    
    def train_svm(self, X_train, y_train):
        """Train SVM model with hyperparameter tuning."""
        if self.models['svm'] is None:
            self.models['svm'] = SVC(random_state=42, probability=True)
            
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto'],
            'decision_function_shape': ['ovr']  # one-vs-rest for multiclass
        }
        
        grid_search = GridSearchCV(self.models['svm'], param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        self.models['svm'] = grid_search.best_estimator_
        print("SVM best parameters:", grid_search.best_params_)
        return grid_search.best_score_
    
    def train_all_models(self, X_train, y_train):
        """Train all models and return their cross-validation scores."""
        scores = {}
        scores['lightgbm'] = self.train_lightgbm(X_train, y_train)
        scores['xgboost'] = self.train_xgboost(X_train, y_train)
        scores['svm'] = self.train_svm(X_train, y_train)
        return scores
    
    def evaluate_model(self, model_name, X_test, y_test):
        """Evaluate a specific model on test data."""
        model = self.models[model_name]
        if model is None:
            raise ValueError(f"Model {model_name} has not been trained yet.")
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'probabilities': y_prob
        }
        return results
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models on test data."""
        results = {}
        for model_name in self.models:
            if self.models[model_name] is not None:
                results[model_name] = self.evaluate_model(model_name, X_test, y_test)
        return results
    
    def save_models(self):
        """Save all trained models to disk."""
        for model_name, model in self.models.items():
            if model is not None:
                save_path = os.path.join(self.save_dir, f"{model_name}_model.joblib")
                joblib.dump(model, save_path)
                print(f"Saved {model_name} model to {save_path}")
    
    def load_models(self):
        """Load all saved models from disk."""
        for model_name in self.models:
            model_path = os.path.join(self.save_dir, f"{model_name}_model.joblib")
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} model from {model_path}")
            else:
                print(f"No saved model found for {model_name}")