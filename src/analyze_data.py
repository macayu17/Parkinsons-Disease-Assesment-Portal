import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor

def analyze_dataset():
    # Load the dataset
    df = pd.read_csv('../PPMI_Curated_Data_Cut_Public_20250714.csv')
    print("\nDataset Shape:", df.shape)

    # Analyze cohort distribution
    print("\nCOHORT Distribution:")
    cohort_dist = df['COHORT'].value_counts()
    print(cohort_dist)

    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Get key features
    key_features = []
    for category in preprocessor.key_features.values():
        key_features.extend(category)
    
    # Analyze key features
    print("\nKey Features Statistics:")
    key_features_stats = df[key_features].describe()
    print(key_features_stats)
    
    # Analyze missing values
    print("\nMissing Values Analysis:")
    missing = df[key_features].isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_summary = pd.DataFrame({
        'Missing Values': missing,
        'Percentage': missing_pct
    }).sort_values('Percentage', ascending=False)
    print(missing_summary[missing_summary['Missing Values'] > 0])

    # Create notebooks directory if it doesn't exist
    import os
    os.makedirs('../notebooks', exist_ok=True)

    # Save visualizations
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='COHORT')
    plt.title('Distribution of Cohorts')
    plt.savefig('../notebooks/cohort_distribution.png')
    plt.close()

    # Correlation analysis for numerical features only
    numerical_features = df[key_features].select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numerical_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('../notebooks/correlation_matrix.png')
    plt.close()

    # Feature distributions
    n_features = len(numerical_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, feature in enumerate(numerical_features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(data=df, x=feature)
        plt.title(feature)
    plt.tight_layout()
    plt.savefig('../notebooks/feature_distributions.png')
    plt.close()

    # Print feature types
    print("\nFeature Types:")
    for category, features in preprocessor.key_features.items():
        print(f"\n{category.title()}:")
        for feature in features:
            if feature in df.columns:
                dtype = df[feature].dtype
                n_unique = df[feature].nunique()
                print(f"- {feature}: {dtype}, {n_unique} unique values")
            else:
                print(f"- {feature}: Not found in dataset")

if __name__ == "__main__":
    analyze_dataset()