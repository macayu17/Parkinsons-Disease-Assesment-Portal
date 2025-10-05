import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.knn_imputer = KNNImputer(n_neighbors=5)
        self.simple_imputer = SimpleImputer(strategy='mean')
        
        # Features to exclude (potential data leakage)
        self.exclude_features = [
            'UPDRS', 'PDTRTMNT', 'LEDD',  # Treatment related
            'MIA_', 'DATSCAN', 'CSF',      # Advanced medical tests
            'Stage_', 'NHY', 'NSD_'        # Disease staging
        ]
        
        # Key features for prediction
        self.key_features = {
            'demographics': ['age', 'SEX', 'EDUCYRS', 'race', 'BMI'],
            'family_history': ['fampd', 'fampd_bin'],
            'symptoms': ['sym_tremor', 'sym_rigid', 'sym_brady', 'sym_posins'],
            'non_motor': ['rem', 'ess', 'gds', 'stai'],
            'cognitive': ['moca', 'clockdraw', 'bjlot']
        }
        
        # Features with high missing rates that need missing indicators
        self.high_missing_features = [
            'sym_tremor', 'sym_rigid', 'sym_brady', 'sym_posins',
            'clockdraw'
        ]
        
        # Features requiring KNN imputation
        self.knn_impute_features = [
            'moca', 'clockdraw', 'bjlot',  # Cognitive scores
            'rem', 'ess', 'gds', 'stai'    # Non-motor scores
        ]
    
    def load_data(self, file_path):
        """Load a single PPMI dataset"""
        try:
            df = pd.read_csv(file_path)
            print(f"Dataset loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
            
    def load_multiple_datasets(self, file_paths):
        """Load and combine multiple PPMI datasets
        
        Args:
            file_paths: List of paths to dataset files
            
        Returns:
            Combined DataFrame with all datasets
        """
        all_dfs = []
        
        for file_path in file_paths:
            try:
                df = pd.read_csv(file_path)
                print(f"Dataset {file_path} loaded successfully with shape: {df.shape}")
                all_dfs.append(df)
            except Exception as e:
                print(f"Error loading dataset {file_path}: {e}")
        
        if not all_dfs:
            print("No datasets were loaded successfully")
            return None
            
        # Combine all datasets
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates if any
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        new_count = len(combined_df)
        
        if original_count > new_count:
            print(f"Removed {original_count - new_count} duplicate records")
            
        print(f"Combined dataset shape: {combined_df.shape}")
        return combined_df
    
    def remove_leakage_features(self, df):
        """Remove features that could cause data leakage"""
        columns_to_drop = []
        for col in df.columns:
            if any(exclude in col for exclude in self.exclude_features):
                columns_to_drop.append(col)
        
        df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Removed {len(columns_to_drop)} potential leakage features")
        return df_cleaned
    
    def select_features(self, df):
        """Select relevant features for prediction"""
        selected_features = []
        for category in self.key_features.values():
            selected_features.extend(category)
        
        # Add target variable
        if 'COHORT' in df.columns:
            selected_features.append('COHORT')
        
        # Keep only available features
        available_features = [f for f in selected_features if f in df.columns]
        print(f"Selected features: {available_features}")
        return df[available_features]
    
    def create_missing_indicators(self, df):
        """Create binary indicators for missing values in high-missing-rate features"""
        for feature in self.high_missing_features:
            if feature in df.columns:
                df[f"{feature}_missing"] = df[feature].isna().astype(int)
        return df
    
    def handle_missing_values(self, df):
        """Enhanced missing value handling with different strategies"""
        # Create missing value indicators
        df = self.create_missing_indicators(df)
        
        # Split features for different imputation strategies
        knn_features = [f for f in self.knn_impute_features if f in df.columns]
        other_numeric = [c for c in df.select_dtypes(include=['float64', 'int64']).columns 
                        if c not in knn_features]
        categorical = df.select_dtypes(include=['object']).columns
        
        # KNN imputation for cognitive and non-motor scores
        if knn_features:
            df[knn_features] = self.knn_imputer.fit_transform(df[knn_features])
        
        # Mean imputation for other numeric features
        if other_numeric:
            df[other_numeric] = self.simple_imputer.fit_transform(df[other_numeric])
        
        # Mode imputation for categorical features
        for col in categorical:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
        
        return df
    
    def engineer_features(self, df):
        """Create engineered features"""
        # Aggregate symptom score
        symptom_cols = [col for col in ['sym_tremor', 'sym_rigid', 'sym_brady', 'sym_posins'] 
                       if col in df.columns]
        if symptom_cols:
            df['total_symptoms'] = df[symptom_cols].sum(axis=1)
        
        # Normalize cognitive scores to 0-1 range
        cognitive_scores = [col for col in ['moca', 'clockdraw', 'bjlot'] 
                          if col in df.columns]
        for col in cognitive_scores:
            df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # Create age groups
        if 'age' in df.columns:
            df['age_group'] = pd.qcut(df['age'], q=5, labels=['VeryYoung', 'Young', 'Middle', 'Old', 'VeryOld'])
        
        # Create BMI categories
        if 'BMI' in df.columns:
            df['bmi_category'] = pd.cut(df['BMI'], 
                                      bins=[0, 18.5, 25, 30, float('inf')],
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        # Interaction features
        if 'age' in df.columns and 'total_symptoms' in df.columns:
            df['age_symptom_interaction'] = df['age'] * df['total_symptoms']
        
        # Cognitive composite score
        cognitive_norm = [col for col in df.columns if col.endswith('_norm')]
        if cognitive_norm:
            df['cognitive_composite'] = df[cognitive_norm].mean(axis=1)
        
        return df
    
    def prepare_data(self, file_path_or_df, test_size=0.2, random_state=42):
        """Complete data preparation pipeline"""
        # Load data
        if isinstance(file_path_or_df, str):
            df = self.load_data(file_path_or_df)
            if df is None:
                return None
        else:
            # Use the provided dataframe directly
            df = file_path_or_df
        
        # Clean and preprocess
        df = self.remove_leakage_features(df)
        df = self.select_features(df)
        df = self.handle_missing_values(df)
        df = self.engineer_features(df)
        
        # Encode all categorical variables (including engineered ones)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            df[col] = self.label_encoder.fit_transform(df[col].astype(str))
        
        # Scale numerical features
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # Split features and target
        if 'COHORT' in df.columns:
            # Encode target variable separately
            y = self.label_encoder.fit_transform(df['COHORT'].astype(str))
            X = df.drop('COHORT', axis=1)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            # Print class distribution
            unique_labels, counts = np.unique(y, return_counts=True)
            print("\nClass distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"Class {label}: {count} samples ({count/len(y)*100:.2f}%)")
            
            return X_train, X_test, y_train, y_test
        else:
            print("Target variable 'COHORT' not found in dataset")
            return None

def main():
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # File path to your dataset
    file_path = "d:\\5th Semester\\Projects\\MiniProject\\Try1\\PPMI_Curated_Data_Cut_Public_20250714.csv"
    
    # Prepare data
    data = preprocessor.prepare_data(file_path)
    
    if data is not None:
        X_train, X_test, y_train, y_test = data
        print("\nData preparation completed successfully!")
        print(f"Training set shape: {X_train.shape}")
        print(f"Testing set shape: {X_test.shape}")

if __name__ == "__main__":
    main()