# Contributing to Parkinson's Disease Assessment Portal

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git installed and configured
- GitHub account
- (Optional) NVIDIA GPU with CUDA support for model training

### Fork and Clone
1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Parkinsons-Disease-Assesment-Portal.git
   cd Parkinsons-Disease-Assesment-Portal
   ```

3. Add upstream remote:
   ```bash
   git remote add upstream https://github.com/macayu17/Parkinsons-Disease-Assesment-Portal.git
   ```

4. Verify remotes:
   ```bash
   git remote -v
   ```
   Should show:
   - `origin` → your fork
   - `upstream` → original repository

## 🌿 Branching Strategy

### Creating a Feature Branch
Always create a new branch from the latest `upstream/main`:

```bash
# Fetch latest changes
git fetch upstream

# Create and switch to new branch
git checkout -b feature/your-feature-name upstream/main
```

### Branch Naming Conventions
- `feature/` - New features (e.g., `feature/add-bert-model`)
- `fix/` - Bug fixes (e.g., `fix/cuda-memory-leak`)
- `docs/` - Documentation updates (e.g., `docs/improve-readme`)
- `refactor/` - Code refactoring (e.g., `refactor/data-preprocessing`)
- `test/` - Adding or improving tests (e.g., `test/model-validation`)

## 💻 Development Workflow

### 1. Set Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### 2. Make Your Changes
- Write clear, readable code
- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Update documentation if needed

### 3. Test Your Changes
```bash
# Run existing tests
python test_system.py

# Test CUDA functionality (if applicable)
python test_cuda.py

# Run specific model tests
cd src
python train_transformer_models.py  # Verify training works
```

### 4. Format and Lint
```bash
# Format code with Black
black src/

# Check for linting issues
flake8 src/

# Type checking (optional)
mypy src/
```

### 5. Commit Your Changes
Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add: Brief description of what you added

- Detailed point 1
- Detailed point 2
- Performance improvements or fixes"
```

**Good commit message examples:**
- `Add: CUDA memory optimization for transformer training`
- `Fix: Handle missing columns in data preprocessing`
- `Docs: Update README with GPU installation instructions`
- `Refactor: Simplify data loading pipeline`

### 6. Push to Your Fork
```bash
git push origin feature/your-feature-name
```

## 📝 Pull Request Process

### Before Submitting
- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation is updated
- [ ] Commit messages are clear
- [ ] Branch is up to date with upstream/main

### Updating Your Branch
If upstream/main has changed since you created your branch:

```bash
# Fetch latest changes
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Force push (if already pushed)
git push origin feature/your-feature-name --force
```

### Creating the Pull Request
1. Go to your fork on GitHub
2. Click "Compare & pull request"
3. **Base repository**: `macayu17/Parkinsons-Disease-Assesment-Portal`
4. **Base branch**: `main`
5. **Head repository**: `YOUR_USERNAME/Parkinsons-Disease-Assesment-Portal`
6. **Compare branch**: `feature/your-feature-name`

### PR Title Format
Use descriptive titles with prefixes:
- `Add: Feature description`
- `Fix: Bug description`
- `Docs: Documentation change`
- `Refactor: Code improvement`

### PR Description Template
```markdown
## Description
Brief description of what this PR does.

## Changes Made
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Tested locally
- [ ] All existing tests pass
- [ ] Added new tests (if applicable)

## Performance Impact
- Training time: X% faster/slower
- Memory usage: X% improvement
- Model accuracy: X% change

## Screenshots (if applicable)
Add screenshots for UI changes.

## Related Issues
Closes #issue_number (if applicable)
```

## 🧪 Testing Guidelines

### Unit Tests
- Add tests for new features
- Ensure existing tests pass
- Test edge cases and error conditions

### Model Training Tests
- Verify models train without errors
- Check model output shapes and types
- Validate accuracy metrics

### Integration Tests
- Test end-to-end workflows
- Verify web interface functionality
- Check RAG system integration

## 📋 Code Style Guidelines

### Python Code Style
- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write descriptive variable names

### Documentation Style
- Use Google-style docstrings
- Include parameter descriptions
- Document return values
- Add usage examples

### Example:
```python
def train_model(data: pd.DataFrame, epochs: int = 10) -> torch.nn.Module:
    """
    Train a transformer model on the provided dataset.
    
    Args:
        data: Preprocessed DataFrame with features and labels
        epochs: Number of training epochs (default: 10)
        
    Returns:
        Trained PyTorch model
        
    Raises:
        ValueError: If data is empty or invalid
        
    Example:
        >>> df = load_data("dataset.csv")
        >>> model = train_model(df, epochs=20)
    """
    # Implementation here
    pass
```

## 🐛 Reporting Issues

### Bug Reports
Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU info)
- Error messages and stack traces

### Feature Requests
Include:
- Clear description of the feature
- Use cases and benefits
- Proposed implementation (if any)
- Examples from other projects (if applicable)

## 🔍 Review Process

### What to Expect
1. Automated checks run on your PR
2. Maintainers review your code
3. Feedback may be provided
4. You may need to make changes
5. Once approved, PR will be merged

### Addressing Feedback
- Be responsive to comments
- Make requested changes promptly
- Push updates to the same branch
- Request re-review when ready

## 📚 Resources

### Documentation
- [README.md](README.md) - Project overview
- [FORK_AND_PR_GUIDE.md](FORK_AND_PR_GUIDE.md) - Detailed fork/PR guide
- [PULL_REQUEST_TEMPLATE.md](PULL_REQUEST_TEMPLATE.md) - PR template

### External Resources
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Python PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## ❓ Questions?

If you have questions:
1. Check existing documentation
2. Search closed issues
3. Open a new issue with the `question` label
4. Join discussions in pull requests

## 🙏 Thank You!

Your contributions make this project better. We appreciate your time and effort!

---

*This contributing guide is maintained by the project team. Last updated: November 2025*
