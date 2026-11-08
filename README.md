# IMOFSE
### Intelligent ML Optimization & Feature Selection Engine

IMOFSE is a machine learning optimization and feature selection project designed to compare multiple classification algorithms and hyperparameter optimization strategies.

## Features

- Multiple ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVC)
- Hyperparameter optimization:
  - GridSearchCV
  - RandomizedSearchCV
- Feature selection using SelectKBest and f_classif
- Stratified K-Fold Cross-Validation
- Multi-metric model evaluation
- Permutation Feature Importance
- Confusion Matrix
- ROC Curve and Precision-Recall Curve
- Learning Curve
- Model comparison and benchmarking
- Model serialization using Joblib
- Automatic experiment reports and visualizations

## Dataset

Breast Cancer Wisconsin Dataset from scikit-learn.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Technologies

Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, SciPy, Joblib.

## Output

The project automatically generates:

- Optimized machine learning model
- Model comparison reports
- Feature selection reports
- Cross-validation results
- Classification reports
- Performance visualizations
- Experiment metadata

All generated files are saved in the same directory as `main.py`.

## Purpose

The purpose of IMOFSE is to demonstrate a structured machine learning workflow involving feature selection, hyperparameter optimization, model evaluation, and experiment tracking.

## Disclaimer

This project uses a public educational dataset and is not intended for clinical diagnosis.
