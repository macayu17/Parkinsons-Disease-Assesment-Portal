# Parkinson's Disease Assessment System

A comprehensive medical assessment system for Parkinson's disease diagnosis using multimodal machine learning and RAG (Retrieval-Augmented Generation).

## 🎯 Features

- **Multimodal Machine Learning**: Combines traditional ML models (LightGBM, XGBoost, SVM) with transformer models for accurate diagnosis
- **GPU Acceleration**: CUDA-enabled PyTorch training with 15x speedup on NVIDIA GPUs
- **High Accuracy**: Achieves 93.19% accuracy with F1 score of 0.9358 using DistilBERT
- **PDF Document Support**: Processes and indexes medical literature in PDF format
- **RAG System**: Retrieves relevant medical literature to enhance diagnostic reports
- **Web Interface**: User-friendly interface for patient data entry and report generation

## System Components

- **Document Manager**: Processes and indexes medical literature
- **Traditional ML Models**: LightGBM, XGBoost, SVM classifiers
- **Transformer Models**: Small, medium, and large transformer models
- **Multimodal Ensemble**: Combines predictions from all models
- **Report Generator**: Creates comprehensive medical reports with literature insights

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Memory**: Minimum 8GB RAM (16GB recommended for model training)

### GPU Requirements (Optional but Recommended)
- **NVIDIA GPU**: Any CUDA-compatible GPU (RTX 3050 or better recommended)
- **CUDA**: Version 12.4 or compatible
- **VRAM**: Minimum 4GB for transformer model training

### Software Dependencies
All Python dependencies are listed in `requirements.txt`, including:
- PyTorch 2.6.0+ with CUDA support
- Transformers 4.34.0
- Scikit-learn, XGBoost, LightGBM
- Flask for web interface
- FAISS for vector similarity search

## 🚀 Installation

### Standard Installation (CPU)
```bash
# Clone the repository
git clone https://github.com/macayu17/Parkinsons-Disease-Assesment-Portal.git
cd Parkinsons-Disease-Assesment-Portal

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Installation (NVIDIA CUDA)
```bash
# Clone the repository
git clone https://github.com/macayu17/Parkinsons-Disease-Assesment-Portal.git
cd Parkinsons-Disease-Assesment-Portal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt

# Verify CUDA setup
python test_cuda.py
```

## Usage

1. Access the web interface at http://localhost:5000
2. Enter patient data in the assessment form
3. View the generated diagnostic report
4. Upload additional medical literature through the documents page

## 🏋️ Model Training

### Quick Start
```bash
cd src

# Train traditional ML models
python train_traditional_models.py

# Train transformer models (GPU recommended)
python train_transformer_models.py

# Train multimodal ensemble
python train_multimodal.py
```

### Training Performance
| Model | Accuracy | F1 Score | Training Time (GPU) | Training Time (CPU) |
|-------|----------|----------|---------------------|---------------------|
| DistilBERT | 93.19% | 0.9358 | ~2 hours | ~30 hours |
| BioBERT | 92.5% | 0.9280 | ~3 hours | ~45 hours |
| PubMedBERT | 91.8% | 0.9210 | ~3 hours | ~45 hours |
| LightGBM | 89.5% | 0.8950 | ~5 minutes | ~5 minutes |
| XGBoost | 88.7% | 0.8870 | ~8 minutes | ~8 minutes |

*GPU benchmarks measured on NVIDIA RTX 3050 Laptop GPU (4GB VRAM)*

### GPU vs CPU Performance
- **Speedup**: 15.32x faster on GPU for transformer models
- **Early Stopping**: Automatically prevents overfitting
- **Model Checkpointing**: Saves best model during training
- **Memory Efficient**: Optimized for 4GB+ VRAM GPUs

## 🧪 Testing

### Verify CUDA Setup
```bash
python test_cuda.py
```

Expected output:
```
✓ CUDA Available: True
✓ CUDA Version: 12.4
✓ GPU: NVIDIA GeForce RTX 3050
✓ GPU Performance: 15.32x faster than CPU
```

### Run System Tests
```bash
python test_system.py
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](FORK_AND_PR_GUIDE.md) for details on:
- Forking the repository
- Creating feature branches
- Submitting pull requests
- Code style guidelines

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

## 🙏 Acknowledgments

- PPMI (Parkinson's Progression Markers Initiative) for the dataset
- Hugging Face for transformer model implementations
- PyTorch team for CUDA optimization support
