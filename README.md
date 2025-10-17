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
4. **Evaluation**: Basic performance metrics (accuracy, precision, recall, F1, ROC-AUC)
5. **Future Work**: Better feature engineering, ensemble methods, hyperparameter tuning

## Baseline Model Results (Midterm)
- **Model Type**: Logistic Regression (simple baseline)
- **Test Accuracy**: ~73-78%
- **F1-Score**: ~75%
- **ROC-AUC**: ~78%
- **Status**: Initial baseline with room for improvement
- **Next Steps**: Better feature engineering, ensemble methods, hyperparameter tuning

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
- Generate feature importance analysis
- Save visualization to `baseline_model_evaluation.png`

### Expected Output
```
Test Accuracy:  ~73-78%
Test F1-Score:  ~75%
Train-Test Gap: ~4-5%
```
