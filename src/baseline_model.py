"""
Smart Ride Baseline Model
Author: Team 5 (Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich)


This baseline model uses Logistic Regression for binary classification to predict
whether a driver should accept (1) or reject (0) a ride request.

Model Details:
- Algorithm: Logistic Regression with L2 regularization
- Input: 32 engineered features
- Output: Binary prediction (accept/reject)
- Performance: ~73-78% accuracy baseline

Features Used:
- Numeric: fare_amount, trip_distance, wait_time, trip_duration, ratings
- Temporal: pickup_hour, day_of_week, is_weekend, is_rush_hour
- Derived: fare_per_km, fare_per_minute, speed_kmh
- Categorical: One-hot encoded payment_type and vehicle_type

Target Variable:
- should_accept: Binary (1 = accept, 0 = reject)
- Based on profitability_score relative to median
- NOTE: profitability_score is EXCLUDED from training to prevent target leakage
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from preprocessing_data import SmartRideRealDataPreprocessor



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
        
        # Train logistic regression with regularization
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=500,
            solver='lbfgs',
            C=0.5  # Stronger regularization for simpler baseline
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        print("  Model training completed")
    
    def evaluate_model(self):
        """
        Basic model evaluation.
        
        Returns:
            Dictionary containing evaluation metrics
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
        
        # Print results
        print("\nACCURACY METRICS:")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy:  {test_accuracy:.4f}")
        print(f"   Accuracy Gap:   {abs(train_accuracy - test_accuracy):.4f}")
        
        print("\nCLASSIFICATION METRICS (Test Set):")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1-Score:  {test_f1:.4f}")
        print(f"   ROC-AUC:   {test_roc_auc:.4f}")
        
        # Interpretation
        train_test_gap = train_accuracy - test_accuracy
        if train_test_gap < 0.05:
            print(f"\n  Good generalization - Small train-test gap ({train_test_gap:.4f})")
        elif train_test_gap < 0.10:
            print(f"\n  Slight overfitting - Train-test gap: {train_test_gap:.4f}")
        else:
            print(f"\n  Overfitting detected - Large gap: {train_test_gap:.4f}")
        
        # Detailed classification report
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
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
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Smart Ride Baseline Model - Performance Evaluation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Performance Metrics Line Plot
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            metrics['test_accuracy'],
            metrics['test_precision'],
            metrics['test_recall'],
            metrics['test_f1'],
            metrics['test_roc_auc']
        ]
        
        # Create line plot with markers
        x_pos = range(len(metric_names))
        axes[0].plot(x_pos, metric_values, marker='o', linewidth=2, markersize=8, 
                     color='#1f77b4', markerfacecolor='#ff7f0e', markeredgecolor='#1f77b4')
        axes[0].set_title('Test Set Performance Metrics')
        axes[0].set_ylim([0, 1])
        axes[0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random Baseline')
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(metric_names, rotation=45)
        axes[0].set_ylabel('Score')
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(x_pos, metric_values)):
            axes[0].text(x, y + 0.02, f'{y:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # 2. Train vs Test Accuracy Comparison Line Plot
        categories = ['Train', 'Test']
        values = [metrics['train_accuracy'], metrics['test_accuracy']]
        
        # Create line plot
        x_pos = range(len(categories))
        axes[1].plot(x_pos, values, marker='s', linewidth=3, markersize=10, 
                     color='#2ca02c', markerfacecolor='#1f77b4', markeredgecolor='#2ca02c')
        axes[1].set_title('Train vs Test Accuracy')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim([min(values) - 0.05, 1.0])
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(categories)
        
        # Add value labels on points
        for i, (x, y) in enumerate(zip(x_pos, values)):
            axes[1].text(x, y + 0.02, f'{y:.4f}', ha='center', va='bottom', fontweight='bold')
        
        gap_text = f"Train-Test Gap: {metrics['variance_indicator']:.4f}"
        axes[1].text(0.5, 0.95, gap_text, transform=axes[1].transAxes,
                       ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n  Visualization saved to: {save_path}")
        
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

    # Load and perform EDA
    print("\n" + "="*70)
    print(" EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    preprocessor = SmartRideRealDataPreprocessor(data_path="data/processed/uber_real_data_processed.csv")
    df = preprocessor.load_data()  # You might have a method for this, or just use pandas.read_csv()
    preprocessor.perform_exploratory_data_analysis(df)

    # Initialize and run baseline model
    baseline = SmartRideBaselineModel(
        data_path="data/processed/uber_real_data_processed_sample.csv"
    )
    
    metrics, feature_importance = baseline.run_baseline_pipeline()
    
    # Summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)
    print(f"\n  Model Type: Logistic Regression (Baseline)")
    print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"  Test F1-Score: {metrics['test_f1']:.4f}")
    print(f"  Train-Test Gap: {metrics['variance_indicator']:.4f}")
    print(f"\nThis baseline establishes a starting point for future model improvements.")
    print("Future work: Better feature engineering, ensemble methods, hyperparameter tuning.\n")


if __name__ == "__main__":
    main()

