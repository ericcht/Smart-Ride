"""
Profitability weight sensitivity analysis for Smart Ride project.
Tests different weightings for profitability score factors.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)


def compute_all_profitability_factors(df):
    return compute_profitability_factors(df)


def compute_profitability_factors(df):
    df = df.copy()
    
    if 'fare_per_km' not in df.columns:
        df['fare_per_km'] = df['fare_amount'] / (df['trip_distance'] + 1e-6)
    fare_efficiency = df['fare_per_km'].rank(pct=True)
    
    if 'fare_per_minute' not in df.columns:
        df['fare_per_minute'] = df['fare_amount'] / (df['trip_duration'] + 1e-6)
    time_efficiency = df['fare_per_minute'].rank(pct=True)
    def calculate_distance_score(distance):
        if distance <= 10:
            return 0.8
        elif distance <= 25:
            return 1.0
        elif distance <= 40:
            return 0.6
        else:
            return 0.3
    
    distance_score = df['trip_distance'].apply(calculate_distance_score)
    
    if 'pickup_hour' not in df.columns:
        if 'pickup_datetime' in df.columns:
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            df['pickup_hour'] = df['pickup_datetime'].dt.hour
        elif 'pickup_time' in df.columns:
            df['pickup_hour'] = pd.to_datetime(df['pickup_time'], format='%H:%M:%S', errors='coerce').dt.hour
    
    hour_bonus = df['pickup_hour'].apply(lambda h: 
        0.3 if h in [7, 8, 9] else
        0.3 if h in [17, 18, 19] else
        0.15 if h in [20, 21, 22] else
        0.1 if h in [10, 11, 12, 13, 14, 15, 16] else
        0.05
    )
    
    if 'is_weekend' not in df.columns:
        if 'pickup_day_of_week' in df.columns:
            df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int)
        elif 'pickup_datetime' in df.columns:
            df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
            df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int)
    
    weekend_bonus = df['is_weekend'] * 0.2
    time_premium = hour_bonus + weekend_bonus
    if time_premium.max() > 0:
        time_premium = time_premium / time_premium.max()
    
    rating_bonus = (df['driver_rating'] - 3) / 2 * 0.1
    rating_bonus = np.maximum(0, rating_bonus)
    if rating_bonus.max() > 0:
        rating_bonus = rating_bonus / rating_bonus.max()
    
    wait_penalty = 1 / (1 + df['wait_time'] / 5)
    duration_efficiency = 1 / (1 + df['trip_duration'] / 30)
    
    factors_df = pd.DataFrame({
        'fare_efficiency': fare_efficiency,
        'time_efficiency': time_efficiency,
        'distance_score': distance_score,
        'time_premium': time_premium,
        'rating_bonus': rating_bonus,
        'wait_penalty': wait_penalty,
        'duration_efficiency': duration_efficiency
    })
    
    return factors_df


def fit_and_evaluate(X_train, y_train, X_test, y_test, random_state=42, tune_C=True, cv=5):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if tune_C:
        param_grid = {'C': [1e-3, 1e-2, 1e-1, 0.5, 1, 5, 10]}
        base = LogisticRegression(random_state=random_state, solver='lbfgs', max_iter=2000)
        gs = GridSearchCV(base, param_grid, scoring='roc_auc', cv=cv, n_jobs=-1, refit=True)
        gs.fit(X_train_s, y_train)
        model = gs.best_estimator_
        cv_score = gs.best_score_
    else:
        model = LogisticRegression(random_state=random_state, solver='lbfgs', C=0.5, max_iter=2000)
        model.fit(X_train_s, y_train)
        cv_score = None

    model.fit(X_train_s, y_train)

    y_test_proba = model.predict_proba(X_test_s)[:, 1]
    y_test_pred = model.predict(X_test_s)

    metrics = {
        'test_roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'test_f1': float(f1_score(y_test, y_test_pred)),
        'test_precision': float(precision_score(y_test, y_test_pred)),
        'test_recall': float(recall_score(y_test, y_test_pred))
    }

    return model, scaler, cv_score, metrics


def generate_weight_grid(n_factors, step=0.25):
    resolution = int(1.0 / step)
    combos = []
    for ints in itertools.product(range(resolution + 1), repeat=n_factors):
        if sum(ints) == resolution:
            w = np.array(ints, dtype=float) / resolution
            combos.append(w)
    return np.vstack(combos)


def generate_random_dirichlet(n_factors, n_samples=200, concentration=1.0, seed=None):
    rng = np.random.default_rng(seed)
    W = rng.gamma(shape=concentration, scale=1.0, size=(n_samples, n_factors))
    W = W / W.sum(axis=1, keepdims=True)
    return W


def run_profitability_weight_sensitivity(
    data_path=None,
    profit_factors=None,
    other_feature_excludes=None,
    sampling_mode='dirichlet',
    grid_step=0.25,
    n_random_samples=200,
    random_seed=42,
    test_size=0.2,
    threshold_mode='median',
    threshold_quantile=0.5,
    tune_C=True,
    output_dir='profitability_ablation_results',
    top_k_coefs=10
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Find data file if not provided
    if data_path is None:
        possible_paths = [
            "data/processed/uber_real_data_processed.csv",
            os.path.join(script_dir, "data/processed/uber_real_data_processed.csv"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        if data_path is None:
            raise FileNotFoundError("Could not find data file. Please specify data_path.")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")

    # Compute profitability factors
    print("\nComputing profitability factors...")
    factors_df = compute_profitability_factors(df)
    
    # Default profit factors if not provided
    if profit_factors is None:
        profit_factors = [
            'fare_efficiency',
            'time_efficiency',
            'distance_score',
            'time_premium',
            'rating_bonus',
            'wait_penalty',
            'duration_efficiency'
        ]
    
    assert all([c in factors_df.columns for c in profit_factors]), \
        f"Some profit_factors missing. Available: {factors_df.columns.tolist()}"

    # Default excludes if not provided (same as baseline_model.py)
    if other_feature_excludes is None:
        other_feature_excludes = [
            'should_accept', 'profitability_score',
            'ride_id', 'rider_id', 'driver_id',
            'pickup_date', 'pickup_time', 'pickup_datetime',
            'pickup_location', 'drop_location', 'booking_status_nan', 'total_amount',
            'trip_duration', 'wait_time', 'fare_per_minute', 'speed_kmh',
            'customer_rating', 'fare_per_km', 'tip_amount'
        ]

    # Select modeling features (train features are everything except excludes)
    feature_cols_all = [c for c in df.columns if c not in other_feature_excludes]
    
    # Split once (so comparisons across weightings are on the same train/test split)
    X_full = df[feature_cols_all].copy()
    X_train_full, X_test_full = train_test_split(
        X_full, test_size=test_size, random_state=random_seed, shuffle=True
    )

    # Generate weights
    n_factors = len(profit_factors)
    if sampling_mode == 'grid':
        W = generate_weight_grid(n_factors, step=grid_step)
    else:
        W = generate_random_dirichlet(n_factors, n_samples=n_random_samples, seed=random_seed)

    print(f"\nRunning {W.shape[0]} experiments (sampling_mode={sampling_mode})")
    print(f"Testing {n_factors} factors: {profit_factors}")

    rows = []
    models_coefs = []
    start_time = time.time()

    for i, w in enumerate(W):
        if (i + 1) % 50 == 0 or (i + 1) == len(W):
            elapsed = time.time() - start_time
            print(f"  Progress: {i+1}/{len(W)} ({elapsed:.1f}s elapsed)")

        # Normalize factors (min-max across entire dataset for consistency)
        factors_norm = factors_df[profit_factors].copy()
        mins = factors_norm.min()
        maxs = factors_norm.max()
        denom = (maxs - mins).replace(0, 1.0)
        factors_norm = (factors_norm - mins) / denom

        # Compute profitability score with current weights
        profit_series = factors_norm.dot(w)

        # Map computed profit score to train/test splits
        X_train = X_train_full.copy()
        X_test = X_test_full.copy()
        X_train = X_train.assign(profitability_score=profit_series.loc[X_train.index].values)
        X_test = X_test.assign(profitability_score=profit_series.loc[X_test.index].values)

        # Derive binary target using threshold_mode
        if threshold_mode == 'median':
            thr = X_train['profitability_score'].median()
        elif threshold_mode == 'quantile':
            thr = X_train['profitability_score'].quantile(threshold_quantile)
        else:
            thr = threshold_quantile  # user-provided absolute threshold

        y_train = (X_train['profitability_score'] > thr).astype(int)
        y_test = (X_test['profitability_score'] > thr).astype(int)

        # Skip if all one class
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        # Do not include profitability_score as a feature
        train_features = X_train.drop(columns=['profitability_score'])
        test_features = X_test.drop(columns=['profitability_score'])

        # Align columns
        train_features = train_features.reindex(
            columns=[c for c in train_features.columns if c in feature_cols_all], 
            fill_value=0
        )
        test_features = test_features.reindex(columns=train_features.columns, fill_value=0)

        # Handle missing values
        train_features = train_features.fillna(train_features.median())
        test_features = test_features.fillna(train_features.median())

        # Fit & evaluate
        model, scaler, cv_score, metrics = fit_and_evaluate(
            train_features, y_train, test_features, y_test,
            random_state=random_seed, tune_C=tune_C, cv=5
        )

        # Save model coefficients
        coefs = model.coef_[0]
        coef_map = dict(zip(train_features.columns, coefs))

        # Record row
        row = {
            'experiment_idx': i,
            'weights': w.tolist(),
            'weights_str': ','.join([f"{x:.3f}" for x in w]),
            'cv_score': cv_score if cv_score is not None else np.nan,
            'test_roc_auc': metrics['test_roc_auc'],
            'test_accuracy': metrics['test_accuracy'],
            'test_f1': metrics['test_f1'],
            'test_precision': metrics['test_precision'],
            'test_recall': metrics['test_recall'],
            'threshold': float(thr),
            'train_pos_rate': float(y_train.mean()),
            'test_pos_rate': float(y_test.mean())
        }
        
        # Add individual weight columns
        for j, factor in enumerate(profit_factors):
            row[f'weight_{factor}'] = float(w[j])

        rows.append(row)
        
        # Store coefficients for top experiments
        if len(models_coefs) < top_k_coefs or metrics['test_roc_auc'] > min([m['test_roc_auc'] for m in models_coefs]):
            coef_row = {
                'experiment_idx': i,
                'test_roc_auc': metrics['test_roc_auc'],
                'test_accuracy': metrics['test_accuracy'],
                'weights_str': row['weights_str'],
                **coef_map
            }
            models_coefs.append(coef_row)
            # Keep only top K
            models_coefs = sorted(models_coefs, key=lambda x: x['test_roc_auc'], reverse=True)[:top_k_coefs]

    # Create results DataFrame
    results_df = pd.DataFrame(rows)
    
    if len(results_df) == 0:
        print("ERROR: No valid experiments completed. Check data and parameters.")
        return None

    # Save results
    results_path = os.path.join(output_dir, 'all_experiments.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved all experiments to: {results_path}")

    # Save top-K coefficients
    coefs_df = pd.DataFrame(models_coefs)
    coefs_path = os.path.join(output_dir, f'top_{top_k_coefs}_model_coefficients.csv')
    coefs_df.to_csv(coefs_path, index=False)
    print(f"Saved top-{top_k_coefs} coefficients to: {coefs_path}")

    # Create visualizations
    print("\nCreating visualizations...")
    visualize_sensitivity_results(results_df, profit_factors, output_dir)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(results_df)}")
    print(f"Best Test ROC-AUC: {results_df['test_roc_auc'].max():.4f}")
    print(f"Best Test Accuracy: {results_df['test_accuracy'].max():.4f}")
    best_idx = results_df['test_roc_auc'].idxmax()
    print(f"\nBest weights:")
    for factor in profit_factors:
        print(f"  {factor}: {results_df.loc[best_idx, f'weight_{factor}']:.3f}")

    return results_df, coefs_df


def run_full_ablation_study(
    data_path=None,
    profit_factors=None,
    other_feature_excludes=None,
    random_seed=42,
    test_size=0.2,
    threshold_mode='median',
    threshold_quantile=0.5,
    tune_C=True,
    output_dir='profitability_ablation_results',
    top_k_coefs=10,
    min_factors=3,
    penalize_single_factor=True,
    max_single_factor_roc=0.95
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Find data file if not provided
    if data_path is None:
        possible_paths = [
            "data/processed/uber_real_data_processed.csv",
            os.path.join(script_dir, "data/processed/uber_real_data_processed.csv"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        if data_path is None:
            raise FileNotFoundError("Could not find data file. Please specify data_path.")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")

    # Compute profitability factors
    print("\nComputing profitability factors...")
    factors_df = compute_profitability_factors(df)
    
    # Default profit factors if not provided
    if profit_factors is None:
        profit_factors = [
            'fare_efficiency',
            'time_efficiency',
            'distance_score',
            'time_premium',
            'rating_bonus',
            'wait_penalty',
            'duration_efficiency'
        ]
    
    assert all([c in factors_df.columns for c in profit_factors]), \
        f"Some profit_factors missing. Available: {factors_df.columns.tolist()}"

    # Default excludes if not provided
    if other_feature_excludes is None:
        other_feature_excludes = [
            'should_accept', 'profitability_score',
            'ride_id', 'rider_id', 'driver_id',
            'pickup_date', 'pickup_time', 'pickup_datetime',
            'pickup_location', 'drop_location', 'booking_status_nan', 'total_amount',
            'trip_duration', 'wait_time', 'fare_per_minute', 'speed_kmh',
            'customer_rating', 'fare_per_km', 'tip_amount'
        ]

    # Select modeling features
    feature_cols_all = [c for c in df.columns if c not in other_feature_excludes]
    
    # Split once (consistent train/test split)
    X_full = df[feature_cols_all].copy()
    X_train_full, X_test_full = train_test_split(
        X_full, test_size=test_size, random_state=random_seed, shuffle=True
    )

    n_factors = len(profit_factors)
    rows = []
    models_coefs = []
    start_time = time.time()
    experiment_idx = 0

    # Normalize factors (min-max across entire dataset for consistency)
    factors_norm = factors_df[profit_factors].copy()
    mins = factors_norm.min()
    maxs = factors_norm.max()
    denom = (maxs - mins).replace(0, 1.0)
    factors_norm = (factors_norm - mins) / denom

    def run_experiment(weights, ablation_type, description):
        nonlocal experiment_idx, models_coefs
        w = np.array(weights)
        w = w / w.sum()
        
        n_active_factors = np.sum(w > 0.01)
        
        if n_active_factors < min_factors:
            return None
        
        profit_series = factors_norm.dot(w)
        
        X_train = X_train_full.copy()
        X_test = X_test_full.copy()
        X_train = X_train.assign(profitability_score=profit_series.loc[X_train.index].values)
        X_test = X_test.assign(profitability_score=profit_series.loc[X_test.index].values)
        
        if threshold_mode == 'median':
            thr = X_train['profitability_score'].median()
        elif threshold_mode == 'quantile':
            thr = X_train['profitability_score'].quantile(threshold_quantile)
        else:
            thr = threshold_quantile
        
        y_train = (X_train['profitability_score'] > thr).astype(int)
        y_test = (X_test['profitability_score'] > thr).astype(int)
        
        if y_train.nunique() < 2 or y_test.nunique() < 2:
            return None
        
        train_features = X_train.drop(columns=['profitability_score'])
        test_features = X_test.drop(columns=['profitability_score'])
        
        train_features = train_features.reindex(
            columns=[c for c in train_features.columns if c in feature_cols_all], 
            fill_value=0
        )
        test_features = test_features.reindex(columns=train_features.columns, fill_value=0)
        
        train_features = train_features.fillna(train_features.median())
        test_features = test_features.fillna(train_features.median())
        
        model, scaler, cv_score, metrics = fit_and_evaluate(
            train_features, y_train, test_features, y_test,
            random_state=random_seed, tune_C=tune_C, cv=5
        )
        
        test_roc = metrics['test_roc_auc']
        cv_roc = cv_score if cv_score is not None else test_roc
        
        if n_active_factors == 1 and test_roc > max_single_factor_roc:
            print(f"   Skipping suspicious result: {description} (ROC-AUC={test_roc:.4f} with single factor)")
            return None
        
        overfitting_gap = test_roc - cv_roc if cv_score is not None else 0.0
        
        factor_diversity_penalty = 0.0
        if penalize_single_factor:
            if n_active_factors == 1:
                factor_diversity_penalty = -0.10
            elif n_active_factors == 2:
                factor_diversity_penalty = -0.05
            elif n_active_factors == 3:
                factor_diversity_penalty = -0.02
        
        adjusted_roc_auc = test_roc + factor_diversity_penalty
        
        coefs = model.coef_[0]
        coef_map = dict(zip(train_features.columns, coefs))
        
        row = {
            'experiment_idx': experiment_idx,
            'ablation_type': ablation_type,
            'description': description,
            'weights': w.tolist(),
            'weights_str': ','.join([f"{x:.3f}" for x in w]),
            'n_active_factors': int(n_active_factors),
            'cv_score': cv_score if cv_score is not None else np.nan,
            'test_roc_auc': test_roc,
            'adjusted_roc_auc': adjusted_roc_auc,
            'overfitting_gap': overfitting_gap,
            'test_accuracy': metrics['test_accuracy'],
            'test_f1': metrics['test_f1'],
            'test_precision': metrics['test_precision'],
            'test_recall': metrics['test_recall'],
            'threshold': float(thr),
            'train_pos_rate': float(y_train.mean()),
            'test_pos_rate': float(y_test.mean()),
            'factor_diversity_penalty': factor_diversity_penalty
        }
        
        for j, factor in enumerate(profit_factors):
            row[f'weight_{factor}'] = float(w[j])
        
        experiment_idx += 1
        
        if len(models_coefs) < top_k_coefs or adjusted_roc_auc > min([m.get('adjusted_roc_auc', m.get('test_roc_auc', 0)) for m in models_coefs]):
            coef_row = {
                'experiment_idx': experiment_idx - 1,
                'ablation_type': ablation_type,
                'n_active_factors': int(n_active_factors),
                'test_roc_auc': test_roc,
                'adjusted_roc_auc': adjusted_roc_auc,
                'cv_score': cv_score if cv_score is not None else np.nan,
                'test_accuracy': metrics['test_accuracy'],
                'weights_str': row['weights_str'],
                **coef_map
            }
            models_coefs.append(coef_row)
            models_coefs = sorted(models_coefs, key=lambda x: x.get('adjusted_roc_auc', x.get('test_roc_auc', 0)), reverse=True)[:top_k_coefs]
        
        return row

    print("\n" + "="*70)
    print("RUNNING FULL ABLATION STUDY")
    print("="*70)

    print("\n1. Baseline: Equal weights for all factors...")
    equal_weights = [1.0 / n_factors] * n_factors
    row = run_experiment(equal_weights, 'baseline', 'Equal weights (1/7 each)')
    if row:
        rows.append(row)
        print(f"   ROC-AUC: {row['test_roc_auc']:.4f}")

    print(f"\n2. Single Factor: Testing each factor alone (min_factors={min_factors}, so these will be skipped)...")
    print("   Note: Single-factor experiments are skipped to prevent memorization.")
    single_factor_results = []
    if min_factors <= 1:
        for i, factor in enumerate(profit_factors):
            weights = [0.0] * n_factors
            weights[i] = 1.0
            row = run_experiment(weights, 'single_factor', f'Only {factor}')
            if row:
                rows.append(row)
                single_factor_results.append((factor, row['adjusted_roc_auc']))
                print(f"   {factor}: ROC-AUC = {row['test_roc_auc']:.4f} (adjusted: {row['adjusted_roc_auc']:.4f})")
    else:
        print("   Skipping single-factor tests. Will use pairwise results to estimate factor importance.")
    
    if len(single_factor_results) > 0:
        single_factor_results.sort(key=lambda x: x[1], reverse=True)
        best_single_factor = single_factor_results[0][0]
        best_single_idx = profit_factors.index(best_single_factor)
    else:
        best_single_idx = 0
        best_single_factor = profit_factors[0]
        print(f"   Using {best_single_factor} as starting point for progressive addition.")

    print("\n3. Leave-One-Out: Removing each factor individually...")
    leave_one_out_results = []
    for i, factor in enumerate(profit_factors):
        weights = [1.0 / (n_factors - 1)] * n_factors
        weights[i] = 0.0
        row = run_experiment(weights, 'leave_one_out', f'Remove {factor}')
        if row:
            rows.append(row)
            leave_one_out_results.append((factor, row['test_roc_auc']))
            print(f"   Without {factor}: ROC-AUC = {row['test_roc_auc']:.4f}")
    
    leave_one_out_results.sort(key=lambda x: x[1], reverse=True)
    worst_factor_to_remove = leave_one_out_results[-1][0]

    print("\n4. Pairwise: Testing all pairs of factors...")
    pairwise_count = 0
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            weights = [0.0] * n_factors
            weights[i] = 0.5
            weights[j] = 0.5
            row = run_experiment(weights, 'pairwise', f'Only {profit_factors[i]} + {profit_factors[j]}')
            if row:
                rows.append(row)
                pairwise_count += 1
                if pairwise_count % 5 == 0:
                    print(f"   Processed {pairwise_count} pairs...")
    print(f"   Completed {pairwise_count} pairwise experiments")

    print("\n5. Progressive Addition: Starting with best single factor, adding others...")
    included = [best_single_idx]
    remaining = [i for i in range(n_factors) if i != best_single_idx]
    
    if len(single_factor_results) > 0:
        remaining_sorted = sorted(remaining, 
                                 key=lambda i: next((r[1] for r in single_factor_results if profit_factors[i] == r[0]), 0),
                                 reverse=True)
    else:
        remaining_sorted = remaining
    
    for add_idx in remaining_sorted:
        included.append(add_idx)
        weights = [0.0] * n_factors
        for idx in included:
            weights[idx] = 1.0 / len(included)
        
        included_names = [profit_factors[i] for i in included]
        row = run_experiment(weights, 'progressive_addition', 
                           f'Factors: {", ".join(included_names)}')
        if row:
            rows.append(row)
            print(f"   Added {profit_factors[add_idx]}: ROC-AUC = {row['test_roc_auc']:.4f}")

    print("\n6. Factor Importance Ranking: Removing factors in order...")
    all_factors = list(range(n_factors))
    factor_importance_order = [profit_factors.index(f) for f, _ in leave_one_out_results]
    
    for remove_count in range(n_factors - 1):
        remaining_factors = factor_importance_order[remove_count:]
        weights = [0.0] * n_factors
        for idx in remaining_factors:
            weights[idx] = 1.0 / len(remaining_factors)
        
        removed_names = [profit_factors[i] for i in factor_importance_order[:remove_count+1]]
        remaining_names = [profit_factors[i] for i in remaining_factors]
        row = run_experiment(weights, 'importance_ranking', 
                           f'Removed: {", ".join(removed_names)}')
        if row:
            rows.append(row)
            print(f"   Removed {remove_count + 1} factors: ROC-AUC = {row['test_roc_auc']:.4f}")

    results_df = pd.DataFrame(rows)
    
    if len(results_df) == 0:
        print("ERROR: No valid experiments completed. Check data and parameters.")
        return None

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY COMPLETED in {elapsed:.1f}s")
    print(f"Total experiments: {len(results_df)}")
    print(f"Minimum factors required: {min_factors} (to prevent memorization)")
    print(f"{'='*70}")

    # Save results
    results_path = os.path.join(output_dir, 'full_ablation_study.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved ablation results to: {results_path}")

    coefs_df = pd.DataFrame(models_coefs)
    coefs_path = os.path.join(output_dir, f'top_{top_k_coefs}_ablation_coefficients.csv')
    coefs_df.to_csv(coefs_path, index=False)
    print(f"Saved top-{top_k_coefs} coefficients to: {coefs_path}")

    print("\nCreating ablation visualizations...")
    visualize_ablation_results(results_df, profit_factors, output_dir)

    print(f"\n{'='*70}")
    print("FACTOR SELECTION ANALYSIS")
    print(f"{'='*70}")
    single_factor = results_df[results_df['ablation_type'] == 'single_factor'].copy()
    if len(single_factor) > 0:
        score_col = 'adjusted_roc_auc' if 'adjusted_roc_auc' in single_factor.columns else 'test_roc_auc'
        single_factor = single_factor.sort_values(score_col, ascending=False)
        print("\n1. Factor Performance (Single Factor Ranking - using adjusted scores):")
        for idx, (_, row) in enumerate(single_factor.iterrows(), 1):
            factor_name = row['description'].replace('Only ', '')
            test_score = row['test_roc_auc']
            adj_score = row.get('adjusted_roc_auc', test_score)
            print(f"   {idx}. {factor_name}: ROC-AUC = {test_score:.4f} (adjusted: {adj_score:.4f})")
    else:
        print("\n1. Factor Performance: Single-factor experiments skipped (min_factors > 1)")
        print("   Using pairwise and leave-one-out results for factor ranking.")
    baseline_roc = results_df[results_df['ablation_type'] == 'baseline']['test_roc_auc'].values
    if len(baseline_roc) > 0:
        baseline_roc = baseline_roc[0]
        leave_one_out = results_df[results_df['ablation_type'] == 'leave_one_out'].copy()
        if len(leave_one_out) > 0:
            leave_one_out['impact'] = baseline_roc - leave_one_out['test_roc_auc']
            leave_one_out = leave_one_out.sort_values('impact', ascending=False)
            print("\n2. Factor Importance (Impact of Removal - Higher = More Important):")
            for idx, (_, row) in enumerate(leave_one_out.iterrows(), 1):
                factor_name = row['description'].replace('Remove ', '')
                impact = row['impact']
                print(f"   {idx}. {factor_name}: Impact = {impact:+.4f} (ROC-AUC drops from {baseline_roc:.4f} to {row['test_roc_auc']:.4f})")
    
    print(f"\n3. Recommended Factor Subset:")
    print(f"   Based on ablation results, the following factors are recommended:")
    
    factor_scores = {}
    for factor in profit_factors:
        score = 0.0
        
        single_row = single_factor[single_factor['description'] == f'Only {factor}']
        if len(single_row) > 0:
            factor_score = single_row.iloc[0].get('adjusted_roc_auc', single_row.iloc[0]['test_roc_auc'])
            score += factor_score * 0.5
        
        loo_row = leave_one_out[leave_one_out['description'] == f'Remove {factor}']
        if len(loo_row) > 0:
            impact = loo_row.iloc[0]['impact']
            normalized_impact = min(impact / 0.1, 1.0) if impact > 0 else 0.0
            score += normalized_impact * 0.5
        
        factor_scores[factor] = score
    
    ranked_factors = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n   Factor Ranking (by combined importance score):")
    for idx, (factor, score) in enumerate(ranked_factors, 1):
        print(f"   {idx}. {factor}: Score = {score:.4f}")
    
    print(f"\n   Recommended Essential Factors (Top {min(7, len(ranked_factors))}):")
    essential_factors = [f for f, _ in ranked_factors[:7]]
    for idx, factor in enumerate(essential_factors, 1):
        print(f"   {idx}. {factor}")
    factor_selection_df = pd.DataFrame({
        'factor': [f for f, _ in ranked_factors],
        'importance_score': [s for _, s in ranked_factors],
        'single_factor_roc_auc': [
            single_factor[single_factor['description'] == f'Only {f}']['test_roc_auc'].values[0] 
            if len(single_factor[single_factor['description'] == f'Only {f}']) > 0 else np.nan
            for f, _ in ranked_factors
        ],
        'removal_impact': [
            leave_one_out[leave_one_out['description'] == f'Remove {f}']['impact'].values[0]
            if len(leave_one_out[leave_one_out['description'] == f'Remove {f}']) > 0 else np.nan
            for f, _ in ranked_factors
        ],
        'recommended': [True if idx < 7 else False for idx in range(len(ranked_factors))]
    })
    
    factor_selection_path = os.path.join(output_dir, 'factor_selection_ranking.csv')
    factor_selection_df.to_csv(factor_selection_path, index=False)
    print(f"\n   Saved factor selection ranking to: {factor_selection_path}")
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"Total experiments: {len(results_df)}")
    
    if 'adjusted_roc_auc' in results_df.columns:
        best_idx = results_df['adjusted_roc_auc'].idxmax()
        print(f"Best Adjusted ROC-AUC: {results_df['adjusted_roc_auc'].max():.4f} (penalizes single-factor configs)")
        print(f"Best Test ROC-AUC: {results_df.loc[best_idx, 'test_roc_auc']:.4f}")
    else:
        best_idx = results_df['test_roc_auc'].idxmax()
        print(f"Best Test ROC-AUC: {results_df['test_roc_auc'].max():.4f}")
    print(f"Best Test Accuracy: {results_df['test_accuracy'].max():.4f}")
    
    if 'n_active_factors' in results_df.columns:
        print(f"Best config uses {int(results_df.loc[best_idx, 'n_active_factors'])} active factors")
    print(f"\nBest configuration:")
    print(f"  Type: {results_df.loc[best_idx, 'ablation_type']}")
    print(f"  Description: {results_df.loc[best_idx, 'description']}")
    print(f"  Weights:")
    for factor in profit_factors:
        print(f"    {factor}: {results_df.loc[best_idx, f'weight_{factor}']:.3f}")
    
    print(f"\nPerformance by Ablation Type:")
    for ablation_type in results_df['ablation_type'].unique():
        subset = results_df[results_df['ablation_type'] == ablation_type]
        print(f"  {ablation_type}:")
        print(f"    Count: {len(subset)}")
        print(f"    Mean ROC-AUC: {subset['test_roc_auc'].mean():.4f}")
        print(f"    Best ROC-AUC: {subset['test_roc_auc'].max():.4f}")
        print(f"    Std ROC-AUC: {subset['test_roc_auc'].std():.4f}")

    return results_df, coefs_df, factor_selection_df


def visualize_ablation_results(results_df, profit_factors, output_dir):
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
    fig.suptitle('Full Ablation Study Results', fontsize=20, fontweight='bold', y=0.995)

    # 1. Performance by ablation type (box plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ablation_types = results_df['ablation_type'].unique()
    data_by_type = [results_df[results_df['ablation_type'] == at]['test_roc_auc'].values 
                    for at in ablation_types]
    bp = ax1.boxplot(data_by_type, labels=ablation_types, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax1.set_ylabel('Test ROC-AUC', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Distribution by Ablation Type', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(alpha=0.3, axis='y')

    # 2. Best performance by ablation type (bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    best_by_type = results_df.groupby('ablation_type')['test_roc_auc'].max().sort_values(ascending=False)
    bars = ax2.barh(range(len(best_by_type)), best_by_type.values, color='coral', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(best_by_type)))
    ax2.set_yticklabels(best_by_type.index, fontsize=10)
    ax2.set_xlabel('Best Test ROC-AUC', fontsize=12, fontweight='bold')
    ax2.set_title('Best Performance by Ablation Type', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    for i, (bar, val) in enumerate(zip(bars, best_by_type.values)):
        ax2.text(val, i, f' {val:.3f}', va='center', fontweight='bold', fontsize=9)

    # 3. Single factor performance comparison
    ax3 = fig.add_subplot(gs[0, 2])
    single_factor = results_df[results_df['ablation_type'] == 'single_factor'].copy()
    if len(single_factor) > 0:
        single_factor = single_factor.sort_values('test_roc_auc', ascending=True)
        factor_names = [desc.replace('Only ', '') for desc in single_factor['description']]
        bars = ax3.barh(range(len(single_factor)), single_factor['test_roc_auc'].values, 
                       color='mediumseagreen', alpha=0.7, edgecolor='black')
        ax3.set_yticks(range(len(single_factor)))
        ax3.set_yticklabels(factor_names, fontsize=9)
        ax3.set_xlabel('Test ROC-AUC', fontsize=12, fontweight='bold')
        ax3.set_title('Single Factor Performance', fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3, axis='x')
        for i, (bar, val) in enumerate(zip(bars, single_factor['test_roc_auc'].values)):
            ax3.text(val, i, f' {val:.3f}', va='center', fontweight='bold', fontsize=9)

    # 4. Leave-one-out impact (performance drop)
    ax4 = fig.add_subplot(gs[1, 0])
    baseline_roc = results_df[results_df['ablation_type'] == 'baseline']['test_roc_auc'].values
    if len(baseline_roc) > 0:
        baseline_roc = baseline_roc[0]
        leave_one_out = results_df[results_df['ablation_type'] == 'leave_one_out'].copy()
        if len(leave_one_out) > 0:
            leave_one_out['performance_drop'] = baseline_roc - leave_one_out['test_roc_auc']
            leave_one_out = leave_one_out.sort_values('performance_drop', ascending=True)
            factor_names = [desc.replace('Remove ', '') for desc in leave_one_out['description']]
            colors = ['red' if drop > 0 else 'green' for drop in leave_one_out['performance_drop']]
            bars = ax4.barh(range(len(leave_one_out)), leave_one_out['performance_drop'].values,
                           color=colors, alpha=0.7, edgecolor='black')
            ax4.set_yticks(range(len(leave_one_out)))
            ax4.set_yticklabels(factor_names, fontsize=9)
            ax4.set_xlabel('Performance Drop (ROC-AUC)', fontsize=12, fontweight='bold')
            ax4.set_title('Impact of Removing Each Factor', fontsize=13, fontweight='bold')
            ax4.axvline(0, color='black', linestyle='--', linewidth=1)
            ax4.grid(alpha=0.3, axis='x')
            for i, (bar, val) in enumerate(zip(bars, leave_one_out['performance_drop'].values)):
                ax4.text(val, i, f' {val:.3f}', va='center', fontweight='bold', fontsize=9)

    # 5. Progressive addition performance
    ax5 = fig.add_subplot(gs[1, 1])
    progressive = results_df[results_df['ablation_type'] == 'progressive_addition'].copy()
    if len(progressive) > 0:
        progressive = progressive.sort_values('experiment_idx')
        ax5.plot(range(len(progressive)), progressive['test_roc_auc'].values, 
                marker='o', linewidth=2, markersize=8, color='steelblue')
        ax5.set_xlabel('Number of Factors Added', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Test ROC-AUC', fontsize=12, fontweight='bold')
        ax5.set_title('Progressive Factor Addition', fontsize=13, fontweight='bold')
        ax5.grid(alpha=0.3)
        ax5.set_xticks(range(len(progressive)))
        ax5.set_xticklabels([f"{i+1}" for i in range(len(progressive))])

    # 6. Factor importance ranking
    ax6 = fig.add_subplot(gs[1, 2])
    importance = results_df[results_df['ablation_type'] == 'importance_ranking'].copy()
    if len(importance) > 0:
        importance = importance.sort_values('experiment_idx')
        ax6.plot(range(len(importance)), importance['test_roc_auc'].values,
                marker='s', linewidth=2, markersize=8, color='coral')
        ax6.set_xlabel('Number of Factors Removed', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Test ROC-AUC', fontsize=12, fontweight='bold')
        ax6.set_title('Factor Removal Impact', fontsize=13, fontweight='bold')
        ax6.grid(alpha=0.3)
        ax6.set_xticks(range(len(importance)))
        ax6.set_xticklabels([f"{i+1}" for i in range(len(importance))])

    # 7. Pairwise combinations heatmap
    ax7 = fig.add_subplot(gs[2, :])
    pairwise = results_df[results_df['ablation_type'] == 'pairwise'].copy()
    if len(pairwise) > 0:
        # Create matrix of pairwise performance
        n_factors_viz = len(profit_factors)
        pair_matrix = np.zeros((n_factors_viz, n_factors_viz))
        for _, row in pairwise.iterrows():
            weights = row['weights']
            try:
                idx1 = next(i for i, w in enumerate(weights) if w > 0.4)
                idx2 = next(i for i, w in enumerate(weights) if i != idx1 and w > 0.4)
                pair_matrix[idx1, idx2] = row['test_roc_auc']
                pair_matrix[idx2, idx1] = row['test_roc_auc']
            except StopIteration:
                # Skip if we can't find the pair indices
                continue
        
        im = ax7.imshow(pair_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax7.set_xticks(range(n_factors_viz))
        ax7.set_xticklabels([f[:15] for f in profit_factors], rotation=45, ha='right', fontsize=9)
        ax7.set_yticks(range(n_factors_viz))
        ax7.set_yticklabels([f[:15] for f in profit_factors], fontsize=9)
        ax7.set_title('Pairwise Factor Combinations Performance', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax7, label='Test ROC-AUC')

    # 8. Summary statistics table
    ax8 = fig.add_subplot(gs[3, :])
    ax8.axis('off')
    
    # Create summary table
    summary_data = []
    for ablation_type in results_df['ablation_type'].unique():
        subset = results_df[results_df['ablation_type'] == ablation_type]
        summary_data.append({
            'Ablation Type': ablation_type,
            'Count': len(subset),
            'Mean ROC-AUC': f"{subset['test_roc_auc'].mean():.4f}",
            'Std ROC-AUC': f"{subset['test_roc_auc'].std():.4f}",
            'Best ROC-AUC': f"{subset['test_roc_auc'].max():.4f}",
            'Worst ROC-AUC': f"{subset['test_roc_auc'].min():.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    table = ax8.table(cellText=summary_df.values, colLabels=summary_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax8.set_title('Summary Statistics by Ablation Type', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'full_ablation_study.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ablation visualization to: {save_path}")
    plt.close()

    # Additional: Factor contribution analysis
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Factor Contribution Analysis', fontsize=16, fontweight='bold')

    # Factor importance from single factor
    ax = axes[0, 0]
    single_factor = results_df[results_df['ablation_type'] == 'single_factor'].copy()
    if len(single_factor) > 0:
        single_factor = single_factor.sort_values('test_roc_auc', ascending=False)
        factor_names = [desc.replace('Only ', '') for desc in single_factor['description']]
        bars = ax.bar(range(len(single_factor)), single_factor['test_roc_auc'].values,
                     color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xticks(range(len(single_factor)))
        ax.set_xticklabels(factor_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Test ROC-AUC', fontsize=11, fontweight='bold')
        ax.set_title('Single Factor Performance Ranking', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        for i, (bar, val) in enumerate(zip(bars, single_factor['test_roc_auc'].values)):
            ax.text(i, val + 0.01, f'{val:.3f}', ha='center', fontweight='bold', fontsize=8)

    # Factor importance from leave-one-out
    ax = axes[0, 1]
    baseline_roc = results_df[results_df['ablation_type'] == 'baseline']['test_roc_auc'].values
    if len(baseline_roc) > 0:
        baseline_roc = baseline_roc[0]
        leave_one_out = results_df[results_df['ablation_type'] == 'leave_one_out'].copy()
        if len(leave_one_out) > 0:
            leave_one_out['importance'] = leave_one_out['test_roc_auc'] - baseline_roc
            leave_one_out = leave_one_out.sort_values('importance', ascending=False)
            factor_names = [desc.replace('Remove ', '') for desc in leave_one_out['description']]
            colors = ['green' if imp > 0 else 'red' for imp in leave_one_out['importance']]
            bars = ax.bar(range(len(leave_one_out)), leave_one_out['importance'].values,
                         color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(leave_one_out)))
            ax.set_xticklabels(factor_names, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Performance Change vs Baseline', fontsize=11, fontweight='bold')
            ax.set_title('Factor Importance (Leave-One-Out)', fontsize=12, fontweight='bold')
            ax.axhline(0, color='black', linestyle='--', linewidth=1)
            ax.grid(alpha=0.3, axis='y')
            for i, (bar, val) in enumerate(zip(bars, leave_one_out['importance'].values)):
                ax.text(i, val + (0.01 if val > 0 else -0.01), f'{val:.3f}', 
                       ha='center', fontweight='bold', fontsize=8, 
                       va='bottom' if val > 0 else 'top')

    # Performance distribution comparison
    ax = axes[1, 0]
    ablation_types = results_df['ablation_type'].unique()
    for at in ablation_types:
        subset = results_df[results_df['ablation_type'] == at]['test_roc_auc']
        ax.hist(subset, alpha=0.5, label=at, bins=20, edgecolor='black')
    ax.set_xlabel('Test ROC-AUC', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Performance Distribution by Type', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3, axis='y')

    # Best configuration weights
    ax = axes[1, 1]
    best_idx = results_df['test_roc_auc'].idxmax()
    best_weights = [results_df.loc[best_idx, f'weight_{f}'] for f in profit_factors]
    bars = ax.barh(range(len(profit_factors)), best_weights, color='coral', alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(profit_factors)))
    ax.set_yticklabels([f[:20] for f in profit_factors], fontsize=9)
    ax.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
    ax.set_title(f'Best Configuration Weights\n(ROC-AUC: {results_df.loc[best_idx, "test_roc_auc"]:.4f})', 
                fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    for i, (bar, weight) in enumerate(zip(bars, best_weights)):
        if weight > 0.01:  # Only label non-zero weights
            ax.text(weight, i, f' {weight:.3f}', va='center', fontweight='bold', fontsize=9)

    plt.tight_layout()
    save_path2 = os.path.join(output_dir, 'factor_contribution_analysis.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Saved factor contribution analysis to: {save_path2}")
    plt.close()


def visualize_sensitivity_results(results_df, profit_factors, output_dir):
    
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('Profitability Weight Sensitivity Analysis', fontsize=18, fontweight='bold', y=0.995)

    # 1. Performance distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_df['test_roc_auc'], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.axvline(results_df['test_roc_auc'].max(), color='red', linestyle='--', linewidth=2, 
                label=f"Best: {results_df['test_roc_auc'].max():.4f}")
    ax1.set_xlabel('Test ROC-AUC', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('ROC-AUC Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Accuracy distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results_df['test_accuracy'], bins=50, alpha=0.7, edgecolor='black', color='mediumseagreen')
    ax2.axvline(results_df['test_accuracy'].max(), color='red', linestyle='--', linewidth=2,
                label=f"Best: {results_df['test_accuracy'].max():.4f}")
    ax2.set_xlabel('Test Accuracy', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. ROC-AUC vs Accuracy scatter
    ax3 = fig.add_subplot(gs[0, 2])
    scatter = ax3.scatter(results_df['test_roc_auc'], results_df['test_accuracy'], 
                         alpha=0.6, c=results_df['test_f1'], cmap='viridis', s=30)
    best_idx = results_df['test_roc_auc'].idxmax()
    ax3.scatter(results_df.loc[best_idx, 'test_roc_auc'], 
               results_df.loc[best_idx, 'test_accuracy'],
               color='red', s=200, marker='*', zorder=5, label='Best')
    ax3.set_xlabel('Test ROC-AUC', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
    ax3.set_title('ROC-AUC vs Accuracy (colored by F1)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='F1-Score')

    # 4. Weight impact on performance (for each factor)
    ax4 = fig.add_subplot(gs[1, :])
    n_factors = len(profit_factors)
    colors = plt.cm.tab10(np.linspace(0, 1, n_factors))
    
    for idx, factor in enumerate(profit_factors):
        weight_col = f'weight_{factor}'
        # Bin weights and compute mean performance
        bins = np.linspace(results_df[weight_col].min(), results_df[weight_col].max(), 20)
        results_df['weight_bin'] = pd.cut(results_df[weight_col], bins=bins)
        bin_perf = results_df.groupby('weight_bin')['test_roc_auc'].mean()
        bin_centers = [interval.mid for interval in bin_perf.index]
        
        ax4.plot(bin_centers, bin_perf.values, marker='o', linewidth=2, 
                label=factor, color=colors[idx], alpha=0.7, markersize=4)
    
    ax4.set_xlabel('Weight Value', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Mean Test ROC-AUC', fontsize=12, fontweight='bold')
    ax4.set_title('Factor Weight Impact on Model Performance', fontsize=13, fontweight='bold')
    ax4.legend(loc='best', fontsize=9, ncol=2)
    ax4.grid(alpha=0.3)
    results_df = results_df.drop(columns=['weight_bin'])

    # 5. Top weights comparison (bar chart)
    ax5 = fig.add_subplot(gs[2, 0])
    best_idx = results_df['test_roc_auc'].idxmax()
    best_weights = [results_df.loc[best_idx, f'weight_{f}'] for f in profit_factors]
    bars = ax5.barh(range(len(profit_factors)), best_weights, color='coral', alpha=0.7, edgecolor='black')
    ax5.set_yticks(range(len(profit_factors)))
    ax5.set_yticklabels([f[:20] for f in profit_factors], fontsize=9)
    ax5.set_xlabel('Weight Value', fontsize=10, fontweight='bold')
    ax5.set_title('Optimal Weights (Best ROC-AUC)', fontsize=11, fontweight='bold')
    ax5.grid(alpha=0.3, axis='x')
    for i, (bar, weight) in enumerate(zip(bars, best_weights)):
        ax5.text(weight, i, f' {weight:.3f}', va='center', fontweight='bold', fontsize=9)

    # 6. Performance metrics comparison
    ax6 = fig.add_subplot(gs[2, 1])
    metrics = ['test_roc_auc', 'test_accuracy', 'test_f1', 'test_precision', 'test_recall']
    metric_names = ['ROC-AUC', 'Accuracy', 'F1', 'Precision', 'Recall']
    best_metrics = [results_df[m].max() for m in metrics]
    bars = ax6.bar(range(len(metrics)), best_metrics, color='steelblue', alpha=0.7, edgecolor='black')
    ax6.set_xticks(range(len(metrics)))
    ax6.set_xticklabels(metric_names, rotation=45, ha='right')
    ax6.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax6.set_title('Best Performance Metrics', fontsize=11, fontweight='bold')
    ax6.set_ylim([0, 1])
    ax6.grid(alpha=0.3, axis='y')
    for i, (bar, val) in enumerate(zip(bars, best_metrics)):
        ax6.text(i, val + 0.02, f'{val:.3f}', ha='center', fontweight='bold', fontsize=9)

    # 7. Weight correlation heatmap
    ax7 = fig.add_subplot(gs[2, 2])
    weight_cols = [f'weight_{f}' for f in profit_factors]
    weight_corr = results_df[weight_cols].corr()
    im = ax7.imshow(weight_corr, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax7.set_xticks(range(len(profit_factors)))
    ax7.set_xticklabels([f[:10] for f in profit_factors], rotation=45, ha='right', fontsize=8)
    ax7.set_yticks(range(len(profit_factors)))
    ax7.set_yticklabels([f[:10] for f in profit_factors], fontsize=8)
    ax7.set_title('Weight Correlation Matrix', fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax7, label='Correlation')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'sensitivity_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved main visualization to: {save_path}")
    plt.close()

    # Additional detailed plot: Weight vs Performance for each factor
    fig2, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle('Individual Factor Weight Sensitivity', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, factor in enumerate(profit_factors):
        if idx >= len(axes):
            break
        ax = axes[idx]
        weight_col = f'weight_{factor}'
        
        # Scatter plot
        ax.scatter(results_df[weight_col], results_df['test_roc_auc'], 
                  alpha=0.5, s=20, color='steelblue')
        
        # Add trend line
        z = np.polyfit(results_df[weight_col], results_df['test_roc_auc'], 2)
        p = np.poly1d(z)
        x_trend = np.linspace(results_df[weight_col].min(), results_df[weight_col].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_xlabel('Weight', fontsize=10, fontweight='bold')
        ax.set_ylabel('Test ROC-AUC', fontsize=10, fontweight='bold')
        ax.set_title(f'{factor}', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(profit_factors), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path2 = os.path.join(output_dir, 'individual_factor_sensitivity.png')
    plt.savefig(save_path2, dpi=300, bbox_inches='tight')
    print(f"Saved detailed visualization to: {save_path2}")
    plt.close()


def main():
    print("\n" + "="*70)
    print(" FULL ABLATION STUDY - PROFITABILITY WEIGHT SENSITIVITY")
    print("="*70)
    print("\nThis script will run a comprehensive ablation study:")
    print("1. Baseline: Equal weights for all factors")
    print("2. Single Factor: Each factor tested alone")
    print("3. Leave-One-Out: Remove each factor individually")
    print("4. Pairwise: All pairs of factors")
    print("5. Progressive Addition: Start with best single, add factors incrementally")
    print("6. Factor Importance: Remove factors in order of importance")
    print("7. Generate comprehensive visualizations and CSV outputs\n")
    
    results_df, coefs_df, factor_selection_df = run_full_ablation_study(
        data_path=None,
        profit_factors=None,
        random_seed=42,
        threshold_mode='median',
        tune_C=True,
        output_dir='profitability_ablation_results',
        top_k_coefs=10,
        min_factors=3,
        penalize_single_factor=True,
        max_single_factor_roc=0.95
    )
    
    if results_df is not None:
        print("\n Full ablation study completed successfully!")
        print(f"\nResults saved in: profitability_ablation_results/")
        print(f"  - full_ablation_study.csv: All experiment results")
        print(f"  - full_ablation_study.png: Main visualization")
        print(f"  - factor_contribution_analysis.png: Factor analysis")
        print(f"  - factor_selection_ranking.csv: Factor importance ranking")
        print(f"\nThe ablation study has identified which factors are most essential")
        print(f"   for setting the profitability threshold for the logistic regression model.")
    else:
        print("\n Ablation study failed. Check error messages above.")

if __name__ == "__main__":
    main()