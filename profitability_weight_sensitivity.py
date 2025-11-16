"""
Smart Ride Profitability Weight Sensitivity Analysis
Author: Team 5 (Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich)

This module performs an ablation/sensitivity study over different weightings used to compute
the "profitability_score" (which determines the label) and evaluates how the downstream
LogisticRegression performance changes. This is the COMPREHENSIVE sensitivity analysis file that includes:
- Profitability factor computation (7 factors: fare efficiency, time efficiency, distance score, etc.)
- Weight sampling strategies (grid search or Dirichlet random sampling)
- Model training and evaluation across multiple weight combinations
- Comprehensive visualization and analysis outputs

Outputs:
- CSV of all experiments (weights, CV score, test ROC-AUC, test accuracy, other metrics)
- CSV of best model coefficients for the top-K experiments
- Plots summarizing sensitivity (performance distributions, weight impact, optimal weights, etc.)

NOTE: This script tests how different weightings of profitability factors affect model performance,
helping to understand the sensitivity of the target variable definition on downstream model quality.
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
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, 'src')
sys.path.insert(0, src_dir)


# ------------------------
# Helper: Compute profitability factors
# ------------------------
def compute_profitability_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 7 profitability factors from raw data columns.
    These are the factors that go into profitability_score.
    
    Returns DataFrame with factor columns added.
    """
    df_factors = df.copy()
    
    # Factor 1: Fare efficiency per km (normalized to percentile)
    if 'fare_per_km' not in df_factors.columns:
        df_factors['fare_per_km'] = df_factors['fare_amount'] / (df_factors['trip_distance'] + 1e-6)
    fare_efficiency = df_factors['fare_per_km'].rank(pct=True)
    
    # Factor 2: Time efficiency per minute (earnings per minute)
    if 'fare_per_minute' not in df_factors.columns:
        df_factors['fare_per_minute'] = df_factors['fare_amount'] / (df_factors['trip_duration'] + 1e-6)
    time_efficiency = df_factors['fare_per_minute'].rank(pct=True)
    
    # Factor 3: Distance preference (categorical scoring)
    def calculate_distance_score(distance):
        if distance <= 10:
            return 0.8  # Short: good turnover
        elif distance <= 25:
            return 1.0  # Medium: optimal range
        elif distance <= 40:
            return 0.6  # Long: acceptable but less ideal
        else:
            return 0.3  # Very long: penalize
    
    distance_score = df_factors['trip_distance'].apply(calculate_distance_score)
    
    # Factor 4: Enhanced time premium (hour-specific bonuses)
    if 'pickup_hour' not in df_factors.columns:
        if 'pickup_datetime' in df_factors.columns:
            df_factors['pickup_datetime'] = pd.to_datetime(df_factors['pickup_datetime'])
            df_factors['pickup_hour'] = df_factors['pickup_datetime'].dt.hour
        elif 'pickup_time' in df_factors.columns:
            # Try to extract hour from time string
            df_factors['pickup_hour'] = pd.to_datetime(df_factors['pickup_time'], format='%H:%M:%S', errors='coerce').dt.hour
    
    hour_bonus = df_factors['pickup_hour'].apply(lambda h: 
        0.3 if h in [7, 8, 9] else  # Morning rush
        0.3 if h in [17, 18, 19] else  # Evening rush
        0.15 if h in [20, 21, 22] else  # Late evening (surge pricing)
        0.1 if h in [10, 11, 12, 13, 14, 15, 16] else  # Daytime
        0.05  # Late night/early morning
    )
    
    if 'is_weekend' not in df_factors.columns:
        if 'pickup_day_of_week' in df_factors.columns:
            df_factors['is_weekend'] = df_factors['pickup_day_of_week'].isin([5, 6]).astype(int)
        elif 'pickup_datetime' in df_factors.columns:
            df_factors['pickup_day_of_week'] = df_factors['pickup_datetime'].dt.dayofweek
            df_factors['is_weekend'] = df_factors['pickup_day_of_week'].isin([5, 6]).astype(int)
    
    weekend_bonus = df_factors['is_weekend'] * 0.2
    time_premium = hour_bonus + weekend_bonus
    # Normalize to 0-1 range
    if time_premium.max() > 0:
        time_premium = time_premium / time_premium.max()
    
    # Factor 5: Rating bonus
    rating_bonus = (df_factors['driver_rating'] - 3) / 2 * 0.1
    rating_bonus = np.maximum(0, rating_bonus)
    # Normalize to 0-1 range
    if rating_bonus.max() > 0:
        rating_bonus = rating_bonus / rating_bonus.max()
    
    # Factor 6: Wait time penalty (shorter is better)
    wait_penalty = 1 / (1 + df_factors['wait_time'] / 5)
    
    # Factor 7: Duration efficiency (penalize very long trips)
    duration_efficiency = 1 / (1 + df_factors['trip_duration'] / 30)
    
    # Create factors DataFrame
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


# ------------------------
# Helper: fit + tune model
# ------------------------
def fit_and_evaluate(X_train, y_train, X_test, y_test, random_state=42, tune_C=True, cv=5):
    """
    Fit logistic regression (optionally tune 'C' via GridSearchCV on X_train).
    Returns fitted model, scaler, CV score (roc_auc), and test metrics dict.
    """
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

    # Ensure final model is fitted on training data
    model.fit(X_train_s, y_train)

    # Predictions / metrics
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


# --------------------------------------------
# Generate weight vectors (simplex) - two modes
# --------------------------------------------
def generate_weight_grid(n_factors: int, step: float=0.25) -> np.ndarray:
    """
    Generate weights on simplex by enumerating non-negative weights that
    sum approximately to 1 with given step.
    Warning: combinatorial explosion if step small or n_factors large.
    Returns shape (n_samples, n_factors).
    """
    resolution = int(1.0 / step)
    combos = []
    for ints in itertools.product(range(resolution + 1), repeat=n_factors):
        if sum(ints) == resolution:
            w = np.array(ints, dtype=float) / resolution
            combos.append(w)
    return np.vstack(combos)


def generate_random_dirichlet(n_factors: int, n_samples: int=200, concentration=1.0, seed=None) -> np.ndarray:
    """Generate random weight vectors from Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    W = rng.gamma(shape=concentration, scale=1.0, size=(n_samples, n_factors))
    W = W / W.sum(axis=1, keepdims=True)
    return W


# -------------------------------------------------------
# Main experiment runner
# -------------------------------------------------------
def run_profitability_weight_sensitivity(
    data_path: str = None,
    profit_factors: List[str] = None,
    other_feature_excludes: List[str] = None,
    sampling_mode: str = 'dirichlet',   # 'grid' or 'dirichlet'
    grid_step: float = 0.25,
    n_random_samples: int = 200,
    random_seed: int = 42,
    test_size: float = 0.2,
    threshold_mode: str = 'median',     # 'median' or 'quantile' or 'absolute'
    threshold_quantile: float = 0.5,
    tune_C: bool = True,
    output_dir: str = 'profitability_ablation_results',
    top_k_coefs: int = 10
):
    """
    Run the sensitivity/ablation study.

    Arguments:
     - data_path: CSV containing preprocessed dataset
     - profit_factors: list of factor names (default: 7 factors from profitability calculation)
     - other_feature_excludes: columns to exclude from features when training
     - sampling_mode: 'grid' enumerates simplex via grid_step, 'dirichlet' samples randomly
     - threshold_mode: how to binarize profitability_score. 'median' will use training-median for each experiment.
     - top_k_coefs: number of top experiments to save coefficients for
    """
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


def visualize_sensitivity_results(results_df: pd.DataFrame, profit_factors: List[str], output_dir: str):
    """Create comprehensive visualizations of sensitivity results."""
    
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
    """Main execution function."""
    print("\n" + "="*70)
    print(" PROFITABILITY WEIGHT SENSITIVITY STUDY")
    print("="*70)
    print("\nThis script will:")
    print("1. Test different weight combinations for profitability factors")
    print("2. Evaluate how each weighting affects model performance")
    print("3. Generate comprehensive visualizations and CSV outputs\n")
    
    # Run the study
    results_df, coefs_df = run_profitability_weight_sensitivity(
        data_path=None,  # Auto-detect
        profit_factors=None,  # Use default 7 factors
        sampling_mode='dirichlet',  # Use random sampling (faster than grid)
        n_random_samples=300,  # Number of weight combinations to test
        random_seed=42,
        threshold_mode='median',  # Use median threshold for binary target
        tune_C=True,  # Tune regularization parameter
        output_dir='profitability_ablation_results',
        top_k_coefs=10
    )
    
    if results_df is not None:
        print("\n✅ Analysis completed successfully!")
        print(f"\nResults saved in: profitability_ablation_results/")
    else:
        print("\n❌ Analysis failed. Check error messages above.")


if __name__ == "__main__":
    main()

