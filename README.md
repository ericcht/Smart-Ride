# Smart Ride: Driver Decision Support

## Project Overview
This project aims to create a machine learning model that helps Uber drivers make informed decisions about whether to accept or decline ride requests by predicting ride profitability.

## Team Members
- Jeffery Chen
- Alex Chen  
- Eric Chtilianov
- Ethan Dietrich

## Problem Statement
Uber drivers often waste time and lose money by accepting unprofitable or inefficient rides. This typically consists of long pickup times, short trip value, and a high chance of cancellation. Our model will act as a decision-support system to predict whether a ride is worth accepting, factoring in distance, time, price, and cancellation patterns.

## Dataset
- **Source**: Uber Ride Analytics Dataset 2024
- **Problem Type**: Binary Classification (Accept vs Reject)
- **Target Variable**: Ride profitability (derived from multiple factors)

## Approach
1. **Data Preprocessing**: Clean and transform raw Uber ride data
2. **Feature Engineering**: Extract temporal, efficiency, and profitability features
3. **Baseline Model**: Logistic Regression for binary classification (Accept/Reject)
4. **Evaluation**: Comprehensive performance metrics including bias-variance analysis
5. **Future Work**: Ensemble methods, deep learning, and real-time optimization

## Baseline Model Results  
- **Model Type**: Logistic Regression
- **Test Accuracy**: 96.50%
- **F1-Score**: 96.55%
- **ROC-AUC**: 99.65%
- **Bias-Variance**: Low variance (σ=0.0165), good generalization (2.5% train-test gap)
- **Top Predictor**: profitability_score (coefficient: +5.03)

## Documentation
- **Baseline Model**: [`docs/BASELINE_MODEL_DOCUMENTATION.md`](docs/BASELINE_MODEL_DOCUMENTATION.md)
- **Feature Engineering**: [`docs/FEATURE_ENGINEERING_DOCUMENTATION.md`](docs/FEATURE_ENGINEERING_DOCUMENTATION.md)

## Project Structure
```
Smart-Ride/
├── data/
│   ├── raw/                                    # Original datasets
│   └── processed/                              # Preprocessed data
├── docs/
│   ├── BASELINE_MODEL_DOCUMENTATION.md         # Model documentation
│   └── FEATURE_ENGINEERING_DOCUMENTATION.md    # Feature engineering docs
├── src/
│   ├── preprocessing_data.py                   # Data cleaning and feature engineering
│   └── baseline_model.py                       # Baseline logistic regression model
├── baseline_model_evaluation.png               # Performance visualization
├── BASELINE_MODEL_SUMMARY.md                   # Model results summary
├── requirements.txt                            # Python dependencies
└── README.md
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Baseline Model
```bash
python src/baseline_model.py
```

This will:
- Load preprocessed data (1,000 samples)
- Train logistic regression model (80/20 split)
- Evaluate with accuracy, precision, recall, F1, ROC-AUC
- Perform bias-variance analysis (5-fold CV)
- Generate feature importance analysis
- Save visualization to `baseline_model_evaluation.png`

### Expected Output
```
Test Accuracy:  96.50%
Test F1-Score:  96.55%
CV Stability:   0.0165 std
Train-Test Gap: 2.50%
```
