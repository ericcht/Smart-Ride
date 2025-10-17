"""
Smart Ride Baseline Model
Author: Team 5 (Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich)

BASELINE MODEL IMPLEMENTATION & DOCUMENTATION
==============================================

TECHNICAL EXPLANATION:
---------------------
This baseline model uses Logistic Regression for binary classification to predict
whether a driver should accept (1) or reject (0) a ride request.

How Logistic Regression Works:
1. Linear Combination: Creates weighted sum of input features
   z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
   
2. Sigmoid Activation: Transforms linear output to probability [0,1]
   P(accept=1|X) = 1 / (1 + e^(-z))
   
3. Decision Boundary: Classifies based on probability threshold (typically 0.5)
   - If P(accept) ≥ 0.5 → Accept ride
   - If P(accept) < 0.5 → Reject ride

4. Training: Minimizes log loss (cross-entropy) using gradient descent
   Loss = -[y log(p) + (1-y) log(1-p)]

Why Logistic Regression as Baseline:
- Simple and interpretable (feature weights show importance)
- Fast training and prediction
- Probabilistic outputs (confidence scores)
- No hyperparameter tuning required for baseline
- Establishes performance floor for complex models

Feature Engineering:
- Numeric features: fare_amount, trip_distance, wait_time, trip_duration, ratings
- Temporal features: pickup_hour, day_of_week, is_weekend, is_rush_hour
- Derived features: fare_per_km, fare_per_minute, speed_kmh
- Categorical features: One-hot encoded payment_type and vehicle_type

Target Variable:
- should_accept: Binary (1 = accept, 0 = reject)
- Based on profitability_score relative to median
- NOTE: profitability_score is EXCLUDED from training features to prevent target leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class SmartRideBaselineModel:
    """
    Baseline model for ride acceptance prediction using Logistic Regression.
    """
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        """
        Initialize baseline model.
        
        Args:
            data_path: Path to preprocessed data
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path or "data/processed/uber_real_data_processed_sample.csv"
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self):
        """
        Load preprocessed data and prepare features/target.
        
        Returns:
            X: Feature matrix
            y: Target vector
        """
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Data shape: {df.shape}")
        print(f"Target distribution:\n{df['should_accept'].value_counts()}")
        
        # Drop non-feature columns
        # IMPORTANT: Exclude profitability_score to avoid target leakage
        exclude_cols = [
            'should_accept', 'profitability_score',
            'ride_id', 'rider_id', 'driver_id',
            'pickup_date', 'pickup_time', 'pickup_datetime',
            'pickup_location', 'drop_location', 'booking_status_nan'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['should_accept'].copy()
        
        # Handle any missing values
        X = X.fillna(X.median())
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        self.feature_columns = feature_cols
        print(f"Number of features: {len(feature_cols)}")
        
        return X, y
    
    def split_data(self, X, y, test_size: float = 0.2):
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"\nData split:")
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Train target distribution:\n{self.y_train.value_counts()}")
        print(f"Test target distribution:\n{self.y_test.value_counts()}")
    
    def train_model(self):
        """
        Train logistic regression baseline model with standardization.
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL (Logistic Regression)")
        print("="*60)
        
        # Standardize features (important for logistic regression)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train logistic regression with L2 regularization (default)
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        print("✓ Model training completed")
    
    def evaluate_model(self):
        """
        Comprehensive model evaluation with bias/variance analysis.
        
        Returns:
            Dictionary containing all evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*60)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_scaled)
        y_test_pred = self.model.predict(self.X_test_scaled)
        y_test_proba = self.model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Calculate metrics
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred)
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        test_roc_auc = roc_auc_score(self.y_test, y_test_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        # Print results
        print("\n1. ACCURACY METRICS:")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy:  {test_accuracy:.4f}")
        print(f"   Accuracy Gap:   {abs(train_accuracy - test_accuracy):.4f}")
        
        print("\n2. CLASSIFICATION METRICS (Test Set):")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1-Score:  {test_f1:.4f}")
        print(f"   ROC-AUC:   {test_roc_auc:.4f}")
        
        print("\n3. CONFUSION MATRIX:")
        print(f"   True Negatives:  {cm[0,0]}")
        print(f"   False Positives: {cm[0,1]}")
        print(f"   False Negatives: {cm[1,0]}")
        print(f"   True Positives:  {cm[1,1]}")
        
        # Bias-Variance Analysis
        print("\n" + "="*60)
        print("BIAS-VARIANCE ANALYSIS")
        print("="*60)
        
        # Cross-validation for variance estimation
        cv_scores = cross_val_score(
            self.model, self.X_train_scaled, self.y_train, 
            cv=5, scoring='accuracy'
        )
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"\n1. VARIANCE ANALYSIS (5-Fold Cross-Validation):")
        print(f"   CV Mean Accuracy: {cv_mean:.4f}")
        print(f"   CV Std Deviation: {cv_std:.4f}")
        print(f"   CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
        
        # Bias analysis
        bias_indicator = 1 - test_accuracy
        train_test_gap = train_accuracy - test_accuracy
        
        print(f"\n2. BIAS ANALYSIS:")
        print(f"   Test Error Rate (Bias Indicator): {bias_indicator:.4f}")
        print(f"   Train-Test Gap (Variance Indicator): {train_test_gap:.4f}")
        
        # Interpretation
        print(f"\n3. INTERPRETATION:")
        if cv_std < 0.02:
            print(f"   ✓ Low variance ({cv_std:.4f}) - Model is stable across folds")
        else:
            print(f"   ⚠ Moderate variance ({cv_std:.4f}) - Some instability")
        
        if train_test_gap < 0.05:
            print(f"   ✓ Good generalization - Small train-test gap ({train_test_gap:.4f})")
        elif train_test_gap < 0.10:
            print(f"   ⚠ Slight overfitting - Train-test gap: {train_test_gap:.4f}")
        else:
            print(f"   ⚠ Overfitting detected - Large gap: {train_test_gap:.4f}")
        
        if test_accuracy > 0.75:
            print(f"   ✓ Good baseline performance ({test_accuracy:.4f})")
        elif test_accuracy > 0.60:
            print(f"   ✓ Acceptable baseline performance ({test_accuracy:.4f})")
        else:
            print(f"   ⚠ Low performance - Consider feature engineering")
        
        # Detailed classification report
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print("\n" + classification_report(
            self.y_test, y_test_pred, 
            target_names=['Reject (0)', 'Accept (1)']
        ))
        
        # Return metrics dictionary
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'confusion_matrix': cm,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'cv_scores': cv_scores,
            'bias_indicator': bias_indicator,
            'variance_indicator': train_test_gap
        }
        
        return metrics
    
    def analyze_feature_importance(self, top_n: int = 15):
        """
        Analyze and display feature importance based on logistic regression coefficients.
        
        Args:
            top_n: Number of top features to display
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get coefficients
        coefficients = self.model.coef_[0]
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 60)
        for idx, row in feature_importance.head(top_n).iterrows():
            direction = "increases" if row['coefficient'] > 0 else "decreases"
            print(f"{row['feature']:30s} | Coef: {row['coefficient']:7.4f} | {direction} acceptance")
        
        return feature_importance
    
    def visualize_results(self, metrics: dict, save_path: str = None):
        """
        Create visualization of model performance.
        
        Args:
            metrics: Dictionary of evaluation metrics
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Smart Ride Baseline Model - Performance Evaluation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['Reject', 'Accept'],
                   yticklabels=['Reject', 'Accept'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        # 2. Performance Metrics Bar Chart
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            metrics['test_accuracy'],
            metrics['test_precision'],
            metrics['test_recall'],
            metrics['test_f1'],
            metrics['test_roc_auc']
        ]
        
        bars = axes[0, 1].bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        axes[0, 1].set_title('Test Set Performance Metrics')
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random Baseline')
        
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.3f}',
                          ha='center', va='bottom', fontweight='bold')
        
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Cross-Validation Scores
        cv_scores = metrics['cv_scores']
        folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
        axes[1, 0].plot(folds, cv_scores, marker='o', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=metrics['cv_mean'], color='green', linestyle='--', 
                          label=f"Mean: {metrics['cv_mean']:.4f}")
        axes[1, 0].axhline(y=metrics['cv_mean'] + metrics['cv_std'], 
                          color='red', linestyle=':', alpha=0.5, label=f"±1 Std")
        axes[1, 0].axhline(y=metrics['cv_mean'] - metrics['cv_std'], 
                          color='red', linestyle=':', alpha=0.5)
        axes[1, 0].set_title('Cross-Validation Accuracy Scores')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim([min(cv_scores) - 0.05, max(cv_scores) + 0.05])
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Bias-Variance Trade-off
        categories = ['Train\nAccuracy', 'Test\nAccuracy', 'CV Mean\nAccuracy']
        values = [metrics['train_accuracy'], metrics['test_accuracy'], metrics['cv_mean']]
        colors_bv = ['#2ca02c', '#1f77b4', '#ff7f0e']
        
        bars_bv = axes[1, 1].bar(categories, values, color=colors_bv)
        axes[1, 1].set_title('Bias-Variance Analysis')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_ylim([min(values) - 0.05, 1.0])
        
        for bar in bars_bv:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.4f}',
                          ha='center', va='bottom', fontweight='bold')
        
        gap_text = f"Train-Test Gap: {metrics['variance_indicator']:.4f}"
        axes[1, 1].text(0.5, 0.95, gap_text, transform=axes[1, 1].transAxes,
                       ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved to: {save_path}")
        
        plt.show()
    
    def run_baseline_pipeline(self):
        """
        Execute complete baseline model pipeline.
        
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*70)
        print(" SMART RIDE BASELINE MODEL - COMPLETE PIPELINE")
        print("="*70)
        
        # Load and prepare data
        X, y = self.load_and_prepare_data()
        
        # Split data
        self.split_data(X, y)
        
        # Train model
        self.train_model()
        
        # Evaluate model
        metrics = self.evaluate_model()
        
        # Feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Visualize results
        self.visualize_results(metrics, save_path='baseline_model_evaluation.png')
        
        print("\n" + "="*70)
        print(" BASELINE MODEL PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return metrics, feature_importance


def main():
    """
    Main execution function for baseline model.
    """
    # Initialize and run baseline model
    baseline = SmartRideBaselineModel(
        data_path="data/processed/uber_real_data_processed_sample.csv"
    )
    
    metrics, feature_importance = baseline.run_baseline_pipeline()
    
    # Summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)
    print(f"\n✓ Model Type: Logistic Regression (Baseline)")
    print(f"✓ Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"✓ Cross-Validation Stability: {metrics['cv_std']:.4f} std")
    print(f"\nThis baseline establishes a performance floor for future models.")
    print("Future improvements: Feature selection, ensemble methods, deep learning.\n")


if __name__ == "__main__":
    main()

