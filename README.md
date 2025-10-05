# Parkinson's Disease Assessment System

A comprehensive medical assessment system for Parkinson's disease diagnosis using multimodal machine learning and RAG (Retrieval-Augmented Generation).

## Features

- **Multimodal Machine Learning**: Combines traditional ML models (LightGBM, XGBoost, SVM) with transformer models for accurate diagnosis
- **PDF Document Support**: Processes and indexes medical literature in PDF format
- **RAG System**: Retrieves relevant medical literature to enhance diagnostic reports
- **Web Interface**: User-friendly interface for patient data entry and report generation

## System Components

- **Document Manager**: Processes and indexes medical literature
- **Traditional ML Models**: LightGBM, XGBoost, SVM classifiers
- **Transformer Models**: Small, medium, and large transformer models
- **Multimodal Ensemble**: Combines predictions from all models
- **Report Generator**: Creates comprehensive medical reports with literature insights

## Installation

```bash
# Clone the repository
git clone https://github.com/macayu17/parkinsons-assessment-system.git

# Install dependencies
pip install -r requirements.txt

# Run the web interface
cd src
python web_interface.py
```

## Usage

1. Access the web interface at http://localhost:5000
2. Enter patient data in the assessment form
3. View the generated diagnostic report
4. Upload additional medical literature through the documents page

## Model Training

The system includes scripts for training all models:

```bash
# Train traditional ML models
python train_traditional_models.py

# Train transformer models
python train_transformer_models.py

# Train multimodal ensemble
python train_multimodal.py
```
