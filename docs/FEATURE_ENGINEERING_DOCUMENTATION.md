# Feature Engineering Documentation

**Team 5**: Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich

---

## Overview

This document describes the feature engineering process for the Smart Ride project, where we transform raw Uber ride data into meaningful features for predicting ride acceptance.

**Input**: 14 raw columns from Kaggle dataset  
**Output**: 32 engineered features  
**Goal**: Predict whether a driver should accept or reject a ride

---

## 1. Feature Extraction

We created new features from the raw data to capture patterns that help predict ride profitability.

### 1.1 Temporal Features (5 features)

Extracted time-based patterns from the pickup datetime:

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `pickup_hour` | Hour of day (0-23) | Captures surge pricing patterns |
| `pickup_day_of_week` | Day of week (0-6) | Monday vs. weekend differences |
| `pickup_month` | Month (1-12) | Seasonal demand variations |
| `is_weekend` | Binary (0/1) | Weekend rides often more profitable |
| `is_rush_hour` | Binary (0/1) | Rush hours (7-9 AM, 5-7 PM) have higher demand |

**Implementation**: Used pandas datetime functions to extract components.

### 1.2 Efficiency Metrics (3 features)

Created ratio-based features to measure ride quality:

| Feature | Formula | Purpose |
|---------|---------|---------|
| `fare_per_km` | fare ÷ distance | Better rate = more profitable |
| `fare_per_minute` | fare ÷ duration | Time efficiency |
| `speed_kmh` | distance ÷ (duration/60) | Traffic conditions indicator |

**Implementation**: Simple division with small epsilon (1e-6) to avoid division by zero.

### 1.3 Profitability Score

Combined multiple factors into a single profitability metric:

```python
profitability_score = 
    fare_efficiency × 0.30 +    # How good is the rate?
    distance_score × 0.25 +      # Is distance optimal (~10km)?
    time_premium × 0.20 +        # Rush hour/weekend bonus
    rating_bonus × 0.15 +        # Driver rating quality
    wait_penalty × 0.10          # Shorter wait = better
```

This score was used to create our target variable but **excluded from training** (see section 2).

### 1.4 Categorical Encoding

Converted text categories into numeric features using one-hot encoding:

- **Payment Type**: Credit Card, Debit Card, UPI, Uber Wallet → 4 binary columns
- **Vehicle Type**: Bike, Mini, Sedan, Premier, XL, eBike → 6 binary columns

**Why one-hot encoding?** Prevents model from assuming categories have an order (e.g., "Credit" isn't "greater than" "Cash").

---

## 2. Feature Selection

We carefully chose which features to include or exclude from model training.

### 2.1 Excluded Features

| Feature | Reason for Exclusion |
|---------|---------------------|
| `profitability_score` | **Target leakage** - used to create the target variable |
| `ride_id`, `rider_id`, `driver_id` | Just identifiers, no predictive value |
| `pickup_date`, `pickup_time`, `pickup_datetime` | Already extracted to temporal features |
| `pickup_location`, `drop_location` | Text data, not encoded yet |

### 2.2 Target Leakage - Why It Matters

**The Problem:**
- We used `profitability_score` to create our target (`should_accept`)
- If we include it as a feature, the model just checks: "Is score > median?"
- This gives artificially high accuracy (~99%) but doesn't learn real patterns

**Our Solution:**
- Exclude `profitability_score` from training features
- Model must learn from observable features only (fare, distance, time, etc.)
- Real accuracy: 96.5% (still excellent, but honest)

### 2.3 Final Feature Set

**32 features used for training:**
- 11 numeric features (fare, distance, ratings, etc.)
- 5 temporal features (hour, day, month, weekend, rush hour)
- 3 efficiency metrics (fare_per_km, fare_per_minute, speed_kmh)
- 13 categorical features (one-hot encoded payment and vehicle types)

---

## 3. Data Preprocessing

Applied standard preprocessing steps before model training:

1. **Missing Values**: Filled with median (more robust than mean)
2. **Infinite Values**: Replaced with median (can occur from divisions)
3. **Standardization**: Scaled all features to mean=0, std=1 (required for logistic regression)
4. **Train-Test Split**: 80% training, 20% testing (stratified to keep class balance)

---

## 4. Feature Importance Results

After training, we found the most important features for predicting ride acceptance:

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

## 5. Results

Our feature engineering contributed significantly to model performance:

- **Raw features only**: ~75-80% accuracy
- **With feature engineering**: 96.5% accuracy
- **Improvement**: ~20% accuracy gain

---

## 6. Code Implementation

**Preprocessing**: `src/preprocessing_data.py`
- Method: `create_profitability_features()` (lines 183-264)

**Feature Selection**: `src/baseline_model.py`
- Method: `load_and_prepare_data()` (lines 84-131)

---

## Conclusion

Effective feature engineering involved:
1. Extracting temporal patterns (hour, day, rush hour)
2. Creating efficiency metrics (fare per km/minute)
3. Encoding categorical variables (payment, vehicle type)
4. Carefully avoiding target leakage (excluding profitability_score)

These steps transformed 14 raw columns into 32 meaningful features, improving model accuracy by ~20%.
