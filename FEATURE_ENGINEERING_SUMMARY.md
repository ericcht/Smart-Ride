# Feature Engineering Summary

**Quick Reference Guide for Smart Ride Feature Engineering**

---

## Feature Extraction Overview

### Input: 14 Raw Columns → Output: 32 Engineered Features

---

## 1. Temporal Features (5 features)

| Feature | Source | Logic |
|---------|--------|-------|
| `pickup_hour` | datetime | Hour of day (0-23) |
| `pickup_day_of_week` | datetime | Day (0=Mon, 6=Sun) |
| `pickup_month` | datetime | Month (1-12) |
| `is_weekend` | day_of_week | 1 if Sat/Sun, 0 otherwise |
| `is_rush_hour` | pickup_hour | 1 if 7-9AM or 5-7PM |

**Impact**: Rush hour (+1.15 coef), Weekend (+0.70 coef)

---

## 2. Derived Efficiency Features (3 features)

| Feature | Formula | Unit |
|---------|---------|------|
| `fare_per_km` | fare ÷ distance | ₹/km |
| `fare_per_minute` | fare ÷ duration | ₹/min |
| `speed_kmh` | distance ÷ (duration/60) | km/h |

**Impact**: 
- fare_per_km: +0.74 coefficient
- speed_kmh: -1.57 coefficient (slower = worse)

---

## 3. Profitability Score (1 feature)

**Formula**:
```
profitability_score = 
  fare_efficiency × 0.30 +
  distance_score × 0.25 +
  time_premium × 0.20 +
  rating_bonus × 0.15 +
  wait_penalty × 0.10
```

**⚠️ EXCLUDED from training** to prevent target leakage!

---

## 4. Categorical Encoding (16 features)

**One-Hot Encoded:**
- `payment_type`: Credit, Debit, UPI, Wallet → 4 binary columns
- `vehicle_type`: Bike, Mini, Sedan, Premier, XL, eBike → 6 binary columns
- Other categoricals: 6 additional binary columns

**Strategy**: Drop first category to avoid multicollinearity

---

## 5. Raw Numeric Features (11 features)

Kept as-is, with standardization:
- fare_amount, trip_distance, wait_time, trip_duration
- driver_rating, customer_rating, tip_amount, total_amount
- Plus coordinates (latitude/longitude)

---

## Feature Selection: What Was Excluded?

| Feature | Reason |
|---------|--------|
| `profitability_score` | **TARGET LEAKAGE** - used to create target |
| `ride_id`, `rider_id`, `driver_id` | Identifiers (no predictive value) |
| `pickup_date`, `pickup_time`, `pickup_datetime` | Extracted to temporal features |
| `pickup_location`, `drop_location` | Text data (future encoding planned) |
| `booking_status_nan` | Single value after filtering |

---

## Critical: Target Leakage Prevention

```
profitability_score → creates → should_accept (target)
         ↓
    EXCLUDED from features ✓
         ↓
    Model learns from observable features only
```

**If included**: 99%+ accuracy (artificial)  
**If excluded**: 96.5% accuracy (real performance) ✓

---

## Preprocessing Pipeline

```
Raw Data
  ↓ Filter (completed rides only)
  ↓ Column mapping
  ↓ Temporal extraction
  ↓ Derived features
  ↓ Categorical encoding
  ↓ Profitability calculation
  ↓ Target creation
  ↓ Feature selection
  ↓ Median imputation (missing values)
  ↓ Z-score standardization
  ↓ 80/20 train-test split (stratified)
  ↓
Model Training
```

---

## Top 10 Feature Importances

| Rank | Feature | Coefficient |
|------|---------|-------------|
| 1 | trip_distance | -1.85 |
| 2 | speed_kmh | -1.57 |
| 3 | fare_amount | +1.34 |
| 4 | total_amount | +1.34 |
| 5 | is_rush_hour | +1.15 |
| 6 | fare_per_km | +0.74 |
| 7 | is_weekend | +0.70 |
| 8 | trip_duration | -0.62 |
| 9 | fare_per_minute | +0.55 |
| 10 | pickup_day_of_week | +0.37 |

**Key Insight**: Distance-fare trade-off dominates decisions

---

## Code References

- **Feature Extraction**: `src/preprocessing_data.py` lines 195-242
- **Feature Selection**: `src/baseline_model.py` lines 97-122
- **Categorical Encoding**: `src/preprocessing_data.py` lines 158-181

---

## Full Documentation

For comprehensive details, see:
- [`docs/FEATURE_ENGINEERING_DOCUMENTATION.md`](docs/FEATURE_ENGINEERING_DOCUMENTATION.md)

---

**Last Updated**: October 17, 2025  
**Team 5**: Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich

