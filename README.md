# Smart Ride: Driver Decision Support

## Overview

Uber drivers often waste time and lose money by accepting unprofitable or inefficient rides in their driving area. This typically consists of long pickup times, short trip value, and a high chance of cancellation. Our model acts as a decision-support system to predict whether a ride is worth accepting, factoring in distance, time, price, and cancellation patterns. This will result in improved Uber drivers' efficiency and earnings.

## Motivation

Driver efficiency and profitability are core metrics for both drivers and rideshare companies:

- **Drivers** can increase their weekly earnings by prioritizing profitable rides
- Understanding which contextual and dynamic features (such as time, distance, surge pricing, and driver performance) impact profitability can inform future pricing and allocation strategies
- By automating this decision process using machine learning, we enable smarter and more equitable ride distribution

## Project Goal

The main goal of this project is to build a reproducible machine learning pipeline that:

1. Processes the real Uber Ride Analytics dataset from Kaggle
2. Engineers features reflecting time, location, and efficiency
3. Selects the most informative features for predictive modeling
4. Predicts ride acceptance decisions based on profitability potential

This pipeline will be the foundation for a driver-assist system recommending whether a trip should be accepted or declined.

## Team Members

- Jeffrey Chen (25% programming, 75% report writing)
- Alex Chen (75% programming, 25% report writing)
- Eric Chtilianov (75% programming, 25% report writing)
- Ethan Dietrich (25% programming, 75% report writing)

## Dataset

### Source

**Dataset**: "Uber Ride Analytics Dashboard" from Kaggle

The dataset contains ride-level data, including:
- Booking value (fare amount)
- Distance
- Duration
- Wait time
- Ratings (driver and customer)
- Payment details
- Vehicle type
- Pickup and drop locations

The dataset was installed programmatically via:
```python
dataset_path = kagglehub.dataset_download("yashdevladdha/uber-ride-analytics-dashboard")
```

### Dataset Statistics

- **Size**: ~93,000 entries with 39 features
- **Fare Amount**: ₹508 on average (≈$5.76 USD), ranging from ₹50 to ₹4,277
- **Trip Distance**: ~26 km per ride (≈16.16 miles), with a maximum of 50 km
- **Wait Time**: ~8.5 minutes on average
- **Trip Duration**: ~30 minutes on average

The Pearson correlation analysis showed weak linear relationships among numeric features, indicating that profitability depends on nonlinear combinations of features such as fare efficiency, time of day, and vehicle type.

## Data Cleaning and Preprocessing

**Module**: `SmartRideRealDataPreprocessor` (`src/preprocessing_data.py`)

### Steps Performed

1. **Column Mapping**: Renamed original dataset columns to our standardized schema:
   ```python
   {
       'Booking ID': 'ride_id',
       'Customer ID': 'rider_id',
       'Booking Value': 'fare_amount',
       'Ride Distance': 'trip_distance',
       'Avg VTAT': 'wait_time',
       'Avg CTAT': 'trip_duration',
       # ... and more
   }
   ```

2. **Filtering**: Filtered for only completed rides with valid fare and distance values

3. **Datetime Conversion**: Combined Date + Time → `pickup_datetime`

4. **Missing Values**: Handled using median imputation

5. **One-Hot Encoding**: Applied to categorical features (`vehicle_type`, `payment_type`, `booking_status`)

6. **Profitability Target Creation**: Derived a custom score based on multiple factors to form the binary variable `should_accept`

## Feature Engineering

**Module**: `SmartRideRealDataPreprocessor.create_profitability_features()`

### Time-Based Features

Extracted from `pickup_datetime`:
- `pickup_hour`: Hour of pickup (0-23)
- `pickup_day_of_week`: Day of week (0-6)
- `pickup_month`: Month of pickup (1-12)
- `is_weekend`: Binary indicator for weekends
- `is_rush_hour`: Binary indicator for rush hours (7-9 AM, 5-7 PM)

### Efficiency Metrics

```python
fare_per_km = fare_amount / (trip_distance + 1e-6)
fare_per_minute = fare_amount / (trip_duration + 1e-6)
speed_kmh = trip_distance / (trip_duration / 60 + 1e-6)
```

Note: Small value (1e-6) added to prevent division by zero.

### Profitability Score

A composite score combining five weighted factors:

| Factor | Description | Weight |
|--------|-------------|--------|
| **Fare Efficiency** | Higher earnings per km ranked as percentile | 0.30 |
| **Distance Preference** | Categorical scoring: Short (0-10 km): 0.8, Medium (10-25 km): 1.0, Long (25-40 km): 0.6, Very Long (40+ km): 0.3 | 0.25 |
| **Time Premium** | Rewards rides during rush hours and weekends | 0.20 |
| **Rating Bonus** | Incentivizes high-rated drivers (service quality proxy) | 0.15 |
| **Wait Time Penalty** | Penalizes long passenger wait times | 0.10 |

**Formula**:
```python
profitability_score = (
    fare_efficiency * 0.3 +
    distance_score * 0.25 +
    time_premium * 0.2 +
    rating_bonus * 0.15 +
    wait_penalty * 0.1
)
```

The binary target is created by comparing the profitability score to the median:
```python
should_accept = (profitability_score > median_profitability).astype(int)
```

## Baseline Model

**Module**: `SmartRideBaselineModel` (`src/baseline_model.py`)

### Model Choice

**Logistic Regression** was chosen as the baseline model due to:
- Simplicity and interpretability
- Suitability for binary classification on tabular data
- Fast training and inference
- Ability to handle regularization

### Model Architecture

- **Algorithm**: Logistic Regression with L2 regularization
- **Regularization Parameter**: C = 0.5 (stronger regularization for baseline)
- **Solver**: 'lbfgs' (efficient for medium-sized datasets)
- **Max Iterations**: 500
- **Features**: 28 engineered features
- **Training Data**: ~74,400 samples (80% split)
- **Test Data**: ~18,600 samples (20% split)

### Training Process

1. **Data Loading**: Loads preprocessed data from `data/processed/uber_real_data_processed.csv`
2. **Feature Selection**: Excludes non-predictive columns (IDs, timestamps) and **profitability_score** (to prevent target leakage)
3. **Data Splitting**: 80/20 train-test split with stratification to preserve class distribution
4. **Feature Scaling**: StandardScaler normalization (critical for Logistic Regression)
5. **Model Training**: Fits Logistic Regression with regularization

### Performance Results

The baseline model achieved excellent performance:

- **Test Accuracy**: 93.5%
- **F1-Score**: 0.935
- **ROC-AUC**: 0.988
- **Train-Test Gap**: 0.003 (excellent generalization, minimal overfitting)

### Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall prediction correctness
- **Precision**: Proportion of predicted accepts that are actually profitable
- **Recall**: Proportion of profitable rides correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

### Feature Importance

The model's coefficients reveal which features most influence acceptance decisions:
- Positive coefficients increase acceptance probability (e.g., higher fare, shorter wait time)
- Negative coefficients decrease acceptance likelihood (e.g., longer distances, longer wait times)

### Model Interpretation

The Logistic Regression model uses:
- **Sigmoid activation function** to compute probabilities
- **0.5 threshold** for binary classification:
  - Probability ≥ 0.5 → Predict 1 (Accept)
  - Probability < 0.5 → Predict 0 (Reject)

## Project Structure

```
Smart-Ride/
├── data/
│   ├── raw/                         # Raw dataset files
│   │   └── uber_data_sample.csv     # Sample data (tracked in git)
│   └── processed/                   # Processed dataset files
│       └── uber_real_data_processed_sample.csv
├── src/
│   ├── preprocessing_data.py        # Data cleaning and preprocessing
│   └── baseline_model.py            # Baseline Logistic Regression model
├── models/                          # Saved trained models (.pkl files)
├── download_data.py                 # Script to download dataset from Kaggle
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Future Work

This baseline establishes a strong starting point for future model improvements:

### Feature Engineering
- Additional temporal indicators (holiday flags, seasonal trends)
- Location-based features (if latitude/longitude data becomes available)
- Interaction features between key variables

### Model Improvements
- **Ensemble Methods**: Random Forests, Gradient Boosting (XGBoost, LightGBM)
- **Alternative Classifiers**: Support Vector Machines, Neural Networks
- **Hyperparameter Tuning**: Grid search and Bayesian optimization

### Evaluation Enhancements
- **Cross-Validation**: k-fold cross-validation for more robust performance estimates
- **Business Metrics**: Simulated driver earnings improvement
- **Fairness Analysis**: Evaluate model performance across different driver segments

## Key Techniques Used

- **Feature Extraction**: Creating derived features from raw data
- **One-Hot Encoding**: Converting categorical variables to numeric
- **Standardization**: Normalizing features for Logistic Regression
- **L2 Regularization**: Preventing overfitting
- **Stratified Sampling**: Preserving class distribution in train-test split
- **Model Evaluation**: Comprehensive metrics for binary classification

## References

- Lecture 05: Learning Theory and Data Preparation (One-hot encoding)
- Lecture 08: Logistic Regression and Decision Trees (Logistic Regression)
- Lecture 09: Bias Variance Regularization and Cross Validation (L2 Regularization)
- Lecture 10: Ensemble Methods (F1 Score, Precision, Accuracy)
- Lecture 13: Dimensionality Reduction Part 2 (Feature Extraction)
