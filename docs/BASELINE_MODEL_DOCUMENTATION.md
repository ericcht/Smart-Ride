# Baseline Model Documentation

**Team 5**: Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich

---

## Overview

This document describes our baseline machine learning model for predicting whether Uber drivers should accept or reject ride requests. We used Logistic Regression as our baseline model.

**Results:**
- **Model**: Logistic Regression
- **Test Accuracy**: 96.50%
- **F1-Score**: 96.55%
- **ROC-AUC**: 99.65%

---

## What is Logistic Regression?

Logistic Regression is a simple machine learning algorithm that predicts binary outcomes (accept/reject rides).

### How It Works:
1. **Input**: Takes features like fare amount, distance, time of day
2. **Calculation**: Combines features with weights to get a score
3. **Probability**: Converts score to probability (0-1)
4. **Decision**: If probability > 0.5 → Accept, else → Reject

### Why We Chose It:
- Simple and easy to understand
- Fast to train and predict
- Provides interpretable results (we can see which features matter most)
- Good baseline for comparison with more complex models

---

## Data and Features

### Input Data
- **Source**: Processed Uber ride data (1,000 rides)
- **Training Set**: 800 rides (80%)
- **Test Set**: 200 rides (20%)
- **Target**: Binary (1 = accept, 0 = reject)

### Features Used (32 total)

**Numeric Features (11):**
- fare_amount, trip_distance, wait_time, trip_duration
- driver_rating, customer_rating, tip_amount, total_amount
- fare_per_km, fare_per_minute, speed_kmh

**Temporal Features (5):**
- pickup_hour, pickup_day_of_week, pickup_month
- is_weekend, is_rush_hour

**Categorical Features (16):**
- Payment types (Credit Card, Debit Card, UPI, Wallet)
- Vehicle types (Bike, Mini, Sedan, Premier, XL, eBike)

**Important**: We excluded `profitability_score` from training features to avoid target leakage (it was used to create our target variable).

---

## Model Performance

### Accuracy Results
| Metric | Train | Test | Gap |
|--------|-------|------|-----|
| **Accuracy** | 99.00% | 96.50% | 2.50% |

**Interpretation**: High test accuracy with small train-test gap means the model generalizes well.

### Classification Metrics (Test Set)
| Metric | Value | Meaning |
|--------|-------|---------|
| **Precision** | 97.03% | Of predicted accepts, 97% were correct |
| **Recall** | 96.08% | Captured 96% of actual accepts |
| **F1-Score** | 96.55% | Balanced precision-recall |
| **ROC-AUC** | 99.65% | Excellent discrimination ability |

### Confusion Matrix
```
                  Predicted
                Reject  Accept
Actual Reject     95      3
       Accept      4     98
```

**Errors**: Only 7 total errors out of 200 test cases (3 false positives, 4 false negatives)

---

## Feature Importance

The most important features for predicting ride acceptance:

| Rank | Feature | Impact | Interpretation |
|------|---------|--------|----------------|
| 1 | trip_distance | -1.85 | Longer trips less desirable |
| 2 | speed_kmh | -1.57 | Slower traffic reduces acceptance |
| 3 | fare_amount | +1.34 | Higher fares increase acceptance |
| 4 | is_rush_hour | +1.15 | Rush hour rides more valuable |
| 5 | fare_per_km | +0.74 | Better rates preferred |
| 6 | is_weekend | +0.70 | Weekend rides more desirable |

**Key Insight**: Drivers prefer high-paying, short-distance rides during peak hours.

---

## Cross-Validation Results

**5-Fold Cross-Validation:**
- **Mean Accuracy**: 97.38%
- **Standard Deviation**: 1.65%
- **Range**: 95.63% - 99.38%

**Interpretation**: Low variance (small standard deviation) means the model is stable and consistent across different data splits.

---

## Model Limitations

1. **Linear Assumption**: Assumes linear relationships between features and outcome
2. **No Feature Interactions**: Doesn't capture complex interactions (e.g., distance × rush_hour)
3. **Limited Data**: Trained on only 1,000 rides
4. **Engineered Target**: Target based on our profitability formula, not real driver decisions

---

## Future Improvements

**Next Steps:**
1. **More Complex Models**: Try Random Forest, XGBoost, or Neural Networks
2. **Feature Engineering**: Add interaction terms, polynomial features
3. **More Data**: Train on larger dataset
4. **Real-Time Features**: Include live traffic, weather data

**Expected Improvements:**
- Random Forest: +2-3% accuracy
- XGBoost: +3-5% accuracy
- Neural Networks: +2-4% accuracy

---

## Conclusion

Our logistic regression baseline achieved:
- 96.5% test accuracy
- Low variance (stable across data splits)
- Interpretable results (clear feature importance)
- Fast training and prediction

This establishes a strong foundation for more advanced models and demonstrates that simple algorithms can work well with good feature engineering.

---

## Technical Details

**Model Configuration:**
```python
LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs'
)
```

**Preprocessing:**
- Missing values filled with median
- Features standardized (mean=0, std=1)
- 80/20 train-test split (stratified)

**Files:**
- Training: `src/baseline_model.py`
- Preprocessing: `src/preprocessing_data.py`
- Data: `data/processed/uber_real_data_processed_sample.csv`