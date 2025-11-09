# Setup Instructions for Medium Data/ML Coding Exercises

## Quick Start

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, imblearn, statsmodels, shap; print('All packages installed!')"
   ```

## Package Descriptions

- **numpy, pandas**: Core data manipulation
- **scikit-learn**: Machine learning algorithms and utilities
- **scipy**: Statistical functions
- **matplotlib**: Plotting and visualization
- **imbalanced-learn**: For Exercise 2 (SMOTE, ADASYN sampling)
- **statsmodels**: For Exercise 3 (ARIMA, time series models)
- **shap**: For Exercise 4 (model interpretability)

## Troubleshooting

### If imbalanced-learn fails to install:
```bash
pip install --upgrade pip setuptools wheel
pip install imbalanced-learn
```

### If statsmodels fails to install:
```bash
pip install --upgrade pip
pip install statsmodels
```

### If SHAP fails to install:
```bash
pip install shap
# On some systems, you may need:
pip install shap --no-deps
pip install numpy scipy scikit-learn pandas
```

## Alternative: Using Conda

If you prefer conda:
```bash
conda create -n ml_medium python=3.10
conda activate ml_medium
conda install numpy pandas scikit-learn scipy matplotlib
pip install imbalanced-learn statsmodels shap
```

