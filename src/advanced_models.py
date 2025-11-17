"""
Smart Ride Advanced Models: Random Forest and Logistic Regression Comparison
Author: Team 5 (Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich)

This module implements Random Forest and compares it with Logistic Regression.
This is the COMPREHENSIVE model file that includes:
- Baseline Logistic Regression (same as baseline_model.py, included for comparison)
- Random Forest Classifier
- Full model comparison and evaluation

NOTE: This file includes the baseline Logistic Regression model, so you don't need
to run baseline_model.py separately. This file trains both models and compares them.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings('ignore')


class SmartRideAdvancedModels:
    """
    Comprehensive model class for ride acceptance prediction.
    
    This class includes:
    - Baseline Logistic Regression (for comparison)
    - Random Forest Classifier
    - Full model comparison and evaluation
    
    This is the main model file - it includes everything you need.
    You don't need to run baseline_model.py separately.
    """
    
    def __init__(self, data_path: str = None, random_state: int = 42):
        """
        Initialize comprehensive model class.
        
        This class trains and compares:
        - Baseline Logistic Regression
        - Random Forest Classifier
        
        Args:
            data_path: Path to preprocessed data
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path or "data/processed/uber_real_data_processed.csv"
        self.random_state = random_state
        self.rf_model = None
        self.baseline_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_explainer_rf = None
        self.shap_explainer_baseline = None
        self.shap_values_rf = None
        self.shap_values_baseline = None
        
    def load_and_prepare_data(self, return_datetime: bool = False):
        """
        Load preprocessed data and prepare features/target.
        Uses the same logic as baseline model for consistency.
        
        Args:
            return_datetime: If True, also return pickup_datetime for time-based splitting
        
        Returns:
            X: Feature matrix
            y: Target vector
            (optional) pickup_datetime: Datetime column for time-based splitting
        """
        print(f"Loading data from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        print(f"Data shape: {df.shape}")
        print(f"Target distribution:\n{df['should_accept'].value_counts()}")
        
        # Store datetime if needed for time-based splitting
        if return_datetime and 'pickup_datetime' in df.columns:
            self.pickup_datetime = pd.to_datetime(df['pickup_datetime'])
        
        # Drop non-feature columns (same exclusions as baseline)
        # IMPORTANT: Exclude profitability_score to avoid target leakage
        exclude_cols = [
        'should_accept', 'profitability_score',
        'ride_id', 'rider_id', 'driver_id',
        'pickup_date', 'pickup_time', 'pickup_datetime',
        'pickup_location', 'drop_location', 'booking_status_nan', 'total_amount',
        'trip_duration', 'wait_time', 'fare_per_minute', 'speed_kmh',
        'customer_rating', 'fare_per_km',
        'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'tip_amount'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df['should_accept'].copy()
        
        # Handle any missing values appropriately by column type
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                # Numeric columns: use median
                X[col] = X[col].fillna(X[col].median())
            else:
                # Boolean/one-hot encoded columns: fill with 0 (no category)
                X[col] = X[col].fillna(0)
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        # Fill infinite values with median for numeric, 0 for boolean
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        self.feature_columns = feature_cols
        print(f"Number of features: {len(feature_cols)}")
        
        if return_datetime and hasattr(self, 'pickup_datetime'):
            return X, y, self.pickup_datetime
        return X, y
    
    def split_data(self, X, y, test_size: float = 0.2, time_based: bool = False, 
                   pickup_datetime: pd.Series = None):
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of test set
            time_based: If True, split by time (train on past, test on future)
            pickup_datetime: Datetime series for time-based splitting
        """
        if time_based and pickup_datetime is not None:
            # Time-based split: train on past, test on future
            # Sort by datetime
            sorted_idx = pickup_datetime.sort_values().index
            X_sorted = X.loc[sorted_idx]
            y_sorted = y.loc[sorted_idx]
            
            # Split at the time threshold
            split_idx = int(len(X_sorted) * (1 - test_size))
            self.X_train = X_sorted.iloc[:split_idx]
            self.X_test = X_sorted.iloc[split_idx:]
            self.y_train = y_sorted.iloc[:split_idx]
            self.y_test = y_sorted.iloc[split_idx:]
            
            print(f"\nTime-based data split (train on past, test on future):")
            print(f"Train period: {pickup_datetime.loc[self.X_train.index].min()} to {pickup_datetime.loc[self.X_train.index].max()}")
            print(f"Test period:  {pickup_datetime.loc[self.X_test.index].min()} to {pickup_datetime.loc[self.X_test.index].max()}")
        else:
            # Random stratified split (default)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state, stratify=y
            )
            print(f"\nRandom stratified data split:")
        
        print(f"Train set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Train target distribution:\n{self.y_train.value_counts()}")
        print(f"Test target distribution:\n{self.y_test.value_counts()}")
    
    def train_random_forest(self, n_estimators: int = 100, max_depth: int = None, 
                           min_samples_split: int = 2, min_samples_leaf: int = 1,
                           max_features: str = 'sqrt'):
        """
        Train Random Forest classifier
        """
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Random Forest doesn't require scaling, but we'll train without it
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=self.random_state,
            n_jobs=-1,  # Use all available cores
            class_weight='balanced',  # Handle class imbalance
            oob_score=True  # Enable out-of-bag scoring for generalization check
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        print(f"  Random Forest training completed")
        print(f"  Number of trees: {n_estimators}")
        print(f"  Max depth: {max_depth if max_depth else 'No limit'}")
        print(f"  Out-of-Bag (OOB) Score: {self.rf_model.oob_score_:.4f}")
        print(f"    (OOB score indicates generalization - should be close to test accuracy)")
    
    def train_baseline(self):
        """
        Train baseline Logistic Regression model.
        
        This trains the same Logistic Regression model as baseline_model.py.
        Used for comparison with Random Forest.
        
        Uses same parameters as baseline_model.py for consistency.
        """
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL (Logistic Regression)")
        print("="*60)
        
        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train logistic regression with same parameters as baseline
        self.baseline_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=500,
            solver='lbfgs',
            C=0.5
        )
        
        self.baseline_model.fit(self.X_train_scaled, self.y_train)
        print("  Baseline Logistic Regression training completed")
    
    def evaluate_model(self, model, model_name: str, use_scaled: bool = False):
        """
        Evaluate a model and return metrics.
        
        Args:
            model: Trained model to evaluate
            model_name: Name of the model for display
            use_scaled: Whether to use scaled features (for Logistic Regression)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        print("\n" + "="*60)
        print(f"MODEL PERFORMANCE EVALUATION: {model_name}")
        print("="*60)
        
        # Prepare data
        if use_scaled:
            X_train_eval = self.X_train_scaled
            X_test_eval = self.X_test_scaled
        else:
            X_train_eval = self.X_train
            X_test_eval = self.X_test
        
        # Predictions
        y_train_pred = model.predict(X_train_eval)
        y_test_pred = model.predict(X_test_eval)
        y_test_proba = model.predict_proba(X_test_eval)[:, 1]
        
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
            'model_name': model_name,
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
        Analyze and display feature importance from Random Forest.
        
        Args:
            top_n: Number of top features to display
        """
        if self.rf_model is None:
            print("Random Forest model not trained yet. Train the model first.")
            return
        
        print("\n" + "="*60)
        print("RANDOM FOREST FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get feature importances
        importances = self.rf_model.feature_importances_
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 60)
        for idx, row in feature_importance.head(top_n).iterrows():
            print(f"{row['feature']:30s} | Importance: {row['importance']:7.4f}")
        
        return feature_importance
    
    def compare_all_models(self):
        """
        Compare all trained models and create comparison visualization.
        
        Returns:
            DataFrame with comparison metrics
        """
        print("\n" + "="*70)
        print(" COMPREHENSIVE MODEL COMPARISON")
        print("="*70)
        
        all_metrics = []
        
        # Evaluate baseline if trained
        if self.baseline_model is not None:
            baseline_metrics = self.evaluate_model(
                self.baseline_model, "Baseline (Logistic Regression)", use_scaled=True
            )
            all_metrics.append(baseline_metrics)
        
        # Evaluate Random Forest
        if self.rf_model is not None:
            rf_metrics = self.evaluate_model(
                self.rf_model, "Random Forest", use_scaled=False
            )
            all_metrics.append(rf_metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics)
        
        print("\n" + "="*70)
        print(" MODEL COMPARISON SUMMARY")
        print("="*70)
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def cross_validate_model(self, model, model_name: str, cv_folds: int = 5, use_scaled: bool = False):
        """
        Perform k-fold cross-validation to assess model generalization.
        
        Args:
            model: Model to cross-validate
            model_name: Name of the model
            cv_folds: Number of cross-validation folds
            use_scaled: Whether to use scaled features
            
        Returns:
            Dictionary with CV results
        """
        print("\n" + "="*60)
        print(f"CROSS-VALIDATION: {model_name}")
        print("="*60)
        
        # Prepare data
        if use_scaled:
            X_data = self.X_train_scaled
        else:
            X_data = self.X_train
        
        # Perform cross-validation
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_accuracy = cross_val_score(model, X_data, self.y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_f1 = cross_val_score(model, X_data, self.y_train, cv=cv, scoring='f1', n_jobs=-1)
        cv_roc_auc = cross_val_score(model, X_data, self.y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        print(f"\n{cv_folds}-Fold Cross-Validation Results:")
        print(f"  Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std() * 2:.4f})")
        print(f"  F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std() * 2:.4f})")
        print(f"  ROC-AUC:  {cv_roc_auc.mean():.4f} (+/- {cv_roc_auc.std() * 2:.4f})")
        
        return {
            'model_name': model_name,
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_roc_auc_mean': cv_roc_auc.mean(),
            'cv_roc_auc_std': cv_roc_auc.std()
        }
    
    def visualize_comparison(self, comparison_df: pd.DataFrame, save_path: str = None):
        """
        Create visualization comparing all models.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Smart Ride - Model Comparison: Logistic Regression vs Random Forest', 
                     fontsize=16, fontweight='bold')
        
        models = comparison_df['model_name'].values
        x_pos = np.arange(len(models))
        
        # 1. Test Accuracy Comparison
        axes[0, 0].bar(x_pos, comparison_df['test_accuracy'], 
                       color=['#1f77b4', '#2ca02c', '#ff7f0e'][:len(models)],
                       alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim([0.85, 1.0])
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 0].grid(alpha=0.3, axis='y')
        for i, v in enumerate(comparison_df['test_accuracy']):
            axes[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. F1-Score Comparison
        axes[0, 1].bar(x_pos, comparison_df['test_f1'], 
                      color=['#1f77b4', '#2ca02c', '#ff7f0e'][:len(models)],
                      alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('F1-Score Comparison')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_ylim([0.85, 1.0])
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(models, rotation=15, ha='right')
        axes[0, 1].grid(alpha=0.3, axis='y')
        for i, v in enumerate(comparison_df['test_f1']):
            axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. ROC-AUC Comparison
        axes[1, 0].bar(x_pos, comparison_df['test_roc_auc'], 
                      color=['#1f77b4', '#2ca02c', '#ff7f0e'][:len(models)],
                      alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('ROC-AUC Comparison')
        axes[1, 0].set_ylabel('ROC-AUC')
        axes[1, 0].set_ylim([0.95, 1.0])
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(models, rotation=15, ha='right')
        axes[1, 0].grid(alpha=0.3, axis='y')
        for i, v in enumerate(comparison_df['test_roc_auc']):
            axes[1, 0].text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Train-Test Gap Comparison (overfitting indicator)
        axes[1, 1].bar(x_pos, comparison_df['variance_indicator'], 
                      color=['#1f77b4', '#2ca02c', '#ff7f0e'][:len(models)],
                      alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Train-Test Gap (Overfitting Indicator)')
        axes[1, 1].set_ylabel('Accuracy Gap')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=15, ha='right')
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
        axes[1, 1].grid(alpha=0.3, axis='y')
        axes[1, 1].legend()
        for i, v in enumerate(comparison_df['variance_indicator']):
            axes[1, 1].text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n  Comparison visualization saved to: {save_path}")
        
        plt.show()
    
    def explain_with_shap(self, model_type: str = 'random_forest', sample_size: int = 100,
                          plot_type: str = 'summary', save_path: str = None, show_plot: bool = False):
        """
        Generate SHAP explanations for model predictions.
        
        Args:
            model_type: 'random_forest' or 'baseline'
            sample_size: Number of samples to use for SHAP calculation (for speed)
            plot_type: 'summary', 'bar', or 'waterfall'
            save_path: Path to save the plot
            show_plot: If True, display plot interactively (default False for speed)
        """
        print("\n" + "="*60)
        print(f"SHAP EXPLAINABILITY ANALYSIS - {model_type.upper()}")
        print("="*60)
        
        if model_type == 'random_forest':
            if self.rf_model is None:
                print("Random Forest model not trained yet.")
                return
            model = self.rf_model
            X_data = self.X_test
            explainer_attr = 'shap_explainer_rf'
            values_attr = 'shap_values_rf'
        elif model_type == 'baseline':
            if self.baseline_model is None:
                print("Baseline model not trained yet.")
                return
            model = self.baseline_model
            X_data = self.X_test_scaled
            explainer_attr = 'shap_explainer_baseline'
            values_attr = 'shap_values_baseline'
        else:
            print(f"Unknown model_type: {model_type}")
            return
        
        # Sample data for faster computation
        if len(X_data) > sample_size:
            sample_indices = np.random.RandomState(self.random_state).choice(
                len(X_data), sample_size, replace=False
            )
            X_sample = X_data.iloc[sample_indices] if hasattr(X_data, 'iloc') else X_data[sample_indices]
        else:
            X_sample = X_data
        
        # Create SHAP explainer if not already created
        if getattr(self, explainer_attr) is None:
            print(f"Creating SHAP explainer for {model_type}...")
            if model_type == 'random_forest':
                # TreeExplainer for tree-based models
                setattr(self, explainer_attr, shap.TreeExplainer(model))
            else:
                # LinearExplainer for linear models
                setattr(self, explainer_attr, shap.LinearExplainer(model, X_data))
        
        explainer = getattr(self, explainer_attr)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # For binary classification, handle both single and double output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Ensure shap_values is numpy array
        shap_values = np.array(shap_values)
        
        # Get actual feature names from sample data
        if hasattr(X_sample, 'columns'):
            actual_feature_names = list(X_sample.columns)
        else:
            actual_feature_names = self.feature_columns
        
        setattr(self, values_attr, shap_values)
        
        # Create visualizations
        print(f"Generating {plot_type} plot...")
        
        if plot_type == 'summary':
            fig = plt.figure(figsize=(14, 10), facecolor='white')
            shap.summary_plot(shap_values, X_sample, 
                            feature_names=actual_feature_names,
                            show=False, max_display=20,
                            plot_size=(14, 10))
            plt.title(f'SHAP Summary Plot - {model_type.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('SHAP Value (impact on prediction)', fontsize=12, fontweight='bold')
            plt.ylabel('Feature', fontsize=12, fontweight='bold')
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout(pad=2)
            
        elif plot_type == 'bar':
            fig = plt.figure(figsize=(12, 10), facecolor='white')
            shap.summary_plot(shap_values, X_sample,
                            feature_names=actual_feature_names,
                            plot_type='bar', show=False, max_display=20)
            plt.title(f'SHAP Feature Importance - {model_type.replace("_", " ").title()}',
                     fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Mean |SHAP value|', fontsize=12, fontweight='bold')
            plt.ylabel('Feature', fontsize=12, fontweight='bold')
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout(pad=2)
            
        elif plot_type == 'waterfall':
            fig = plt.figure(figsize=(12, 10), facecolor='white')
            if model_type == 'random_forest':
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                expected_value = explainer.expected_value
            
            # Ensure expected_value is scalar
            if hasattr(expected_value, '__len__') and len(expected_value) > 0:
                expected_value = float(expected_value[0])
            else:
                expected_value = float(expected_value)

            # Prepare single-sample SHAP values and features with aligned length
            single_shap = np.array(shap_values[0]).flatten()
            if hasattr(X_sample, 'iloc'):
                single_features = X_sample.iloc[0].values
            else:
                single_features = np.array(X_sample[0]).flatten()

            n_features = len(actual_feature_names)
            n_shap = len(single_shap)
            n_vals = len(single_features)
            min_len = min(n_features, n_shap, n_vals)
            if min_len == 0:
                print("Warning: No features to display for waterfall plot")
            else:
                actual_feature_names_local = actual_feature_names[:min_len]
                single_shap = single_shap[:min_len]
                single_features = single_features[:min_len]

                shap.waterfall_plot(
                    shap.Explanation(
                        values=single_shap,
                        base_values=expected_value,
                        data=single_features,
                        feature_names=actual_feature_names_local
                    ),
                    show=False,
                    max_display=15
                )
            plt.title(f'SHAP Waterfall Plot (First Sample) - {model_type.replace("_", " ").title()}',
                     fontsize=16, fontweight='bold', pad=20)
            plt.tight_layout(pad=2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP plot saved to: {save_path}")
        
        if show_plot:
            plt.show()
        plt.close('all')
        
        # Print top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        # Ensure 1D array
        if len(mean_abs_shap.shape) > 1:
            mean_abs_shap = mean_abs_shap.flatten()
        
        # Ensure arrays match in length
        n_features = len(actual_feature_names)
        n_shap = len(mean_abs_shap)
        
        if n_shap != n_features:
            # Take only the matching number of features
            min_len = min(n_features, n_shap)
            feature_importance_shap = pd.DataFrame({
                'feature': actual_feature_names[:min_len],
                'mean_abs_shap': mean_abs_shap[:min_len]
            }).sort_values('mean_abs_shap', ascending=False)
            print(f"Warning: Feature count mismatch (features: {n_features}, SHAP: {n_shap}). Using first {min_len}.")
        else:
            feature_importance_shap = pd.DataFrame({
                'feature': actual_feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values('mean_abs_shap', ascending=False)
        
        print(f"\nTop 15 Most Important Features (by mean |SHAP value|):")
        print("-" * 60)
        for idx, row in feature_importance_shap.head(15).iterrows():
            print(f"{row['feature']:30s} | SHAP: {row['mean_abs_shap']:7.4f}")
        
        return shap_values, feature_importance_shap
    
    def explain_single_prediction(self, sample_index: int, model_type: str = 'random_forest',
                                  save_path: str = None, show_plot: bool = False):
        """
        Explain a single prediction using SHAP waterfall plot.
        
        Args:
            sample_index: Index of the sample to explain
            model_type: 'random_forest' or 'baseline'
            save_path: Path to save the plot
            show_plot: If True, display plot interactively (default False for speed)
        """
        print("\n" + "="*60)
        print(f"SINGLE PREDICTION EXPLANATION - {model_type.upper()}")
        print("="*60)
        
        if model_type == 'random_forest':
            if self.rf_model is None:
                print("Random Forest model not trained yet.")
                return
            model = self.rf_model
            X_data = self.X_test
            explainer_attr = 'shap_explainer_rf'
        else:
            if self.baseline_model is None:
                print("Baseline model not trained yet.")
                return
            model = self.baseline_model
            X_data = self.X_test_scaled
            explainer_attr = 'shap_explainer_baseline'
        
        # Create explainer if needed
        if getattr(self, explainer_attr) is None:
            print(f"Creating SHAP explainer for {model_type}...")
            if model_type == 'random_forest':
                setattr(self, explainer_attr, shap.TreeExplainer(model))
            else:
                setattr(self, explainer_attr, shap.LinearExplainer(model, X_data))
        
        explainer = getattr(self, explainer_attr)
        
        # Get single sample
        X_single = X_data.iloc[sample_index:sample_index+1]
        
        # Get actual feature names from data
        if hasattr(X_single, 'columns'):
            actual_feature_names = list(X_single.columns)
        else:
            actual_feature_names = self.feature_columns
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_single)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Ensure numpy array
        shap_values = np.array(shap_values)
        
        # Get prediction
        if model_type == 'random_forest':
            prediction = model.predict(X_single)[0]
            proba = model.predict_proba(X_single)[0]
        else:
            prediction = model.predict(X_single)[0]
            proba = model.predict_proba(X_single)[0]
        
        print(f"\nSample Index: {sample_index}")
        print(f"Prediction: {prediction} ({'Accept' if prediction == 1 else 'Reject'})")
        print(f"Probability: Reject={proba[0]:.4f}, Accept={proba[1]:.4f}")
        print(f"Actual: {self.y_test.iloc[sample_index]}")
        
        # Handle shap_values shape and prepare variables FIRST
        if len(shap_values.shape) > 1:
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
        
        shap_vals = np.array(shap_vals).flatten()
        
        if hasattr(X_single, 'iloc'):
            feature_vals = X_single.iloc[0].values
        else:
            feature_vals = np.array(X_single[0]).flatten()
        
        # Ensure arrays match in length BEFORE plotting
        n_features = len(actual_feature_names)
        n_shap = len(shap_vals)
        n_vals = len(feature_vals)
        min_len = min(n_features, n_shap, n_vals)
        if min_len == 0:
            print("Warning: No features to display")
            return
        
        # Truncate to common length so SHAP waterfall does not index out of bounds
        actual_feature_names = actual_feature_names[:min_len]
        shap_vals = shap_vals[:min_len]
        feature_vals = feature_vals[:min_len]
        
        # Create waterfall plot
        if model_type == 'random_forest':
            expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            expected_value = explainer.expected_value
        
        # Ensure expected_value is scalar
        if hasattr(expected_value, '__len__') and len(expected_value) > 0:
            expected_value = float(expected_value[0])
        else:
            expected_value = float(expected_value)
        
        fig = plt.figure(figsize=(12, 10), facecolor='white')
        
        explanation = shap.Explanation(
            values=shap_vals,
            base_values=expected_value,
            data=feature_vals,
            feature_names=actual_feature_names
        )
        
        shap.waterfall_plot(explanation, show=False, max_display=15)
        
        plt.title(f'SHAP Explanation - Sample {sample_index} - {model_type.replace("_", " ").title()}',
                 fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout(pad=2)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP explanation saved to: {save_path}")
        
        if show_plot:
            plt.show()
        plt.close('all')
        
        # Print feature contributions
        feature_contributions = pd.DataFrame({
            'feature': actual_feature_names,
            'shap_value': shap_vals,
            'feature_value': feature_vals
        }).sort_values('shap_value', key=abs, ascending=False)
        
        print(f"\nTop 10 Feature Contributions:")
        print("-" * 70)
        print(f"{'Feature':<30s} | {'Value':<10s} | {'SHAP':<10s}")
        print("-" * 70)
        for idx, row in feature_contributions.head(10).iterrows():
            print(f"{row['feature']:<30s} | {row['feature_value']:<10.4f} | {row['shap_value']:<10.4f}")
    
    def get_prediction_confidence_intervals(self, model_type: str = 'random_forest',
                                           confidence_level: float = 0.95,
                                           n_bootstrap: int = 100):
        """
        Calculate confidence intervals for predictions using bootstrap.
        
        Args:
            model_type: 'random_forest' or 'baseline'
            confidence_level: Confidence level (default 0.95 for 95% CI)
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            DataFrame with predictions and confidence intervals
        """
        print("\n" + "="*60)
        print(f"PREDICTION CONFIDENCE INTERVALS - {model_type.upper()}")
        print("="*60)
        
        if model_type == 'random_forest':
            if self.rf_model is None:
                print("Random Forest model not trained yet.")
                return
            model = self.rf_model
            X_data = self.X_test
            use_scaled = False
        else:
            if self.baseline_model is None:
                print("Baseline model not trained yet.")
                return
            model = self.baseline_model
            X_data = self.X_test_scaled
            use_scaled = True
        
        print(f"Calculating confidence intervals using {n_bootstrap} bootstrap samples...")
        
        # Get base predictions
        y_pred_proba = model.predict_proba(X_data)[:, 1]
        
        # Bootstrap for confidence intervals
        bootstrap_predictions = np.zeros((len(X_data), n_bootstrap))
        
        for i in range(n_bootstrap):
            # Resample training data
            sample_indices = np.random.RandomState(self.random_state + i).choice(
                len(self.X_train), len(self.X_train), replace=True
            )
            
            if use_scaled:
                X_train_boot = self.X_train_scaled[sample_indices]
                y_train_boot = self.y_train.iloc[sample_indices]
            else:
                X_train_boot = self.X_train.iloc[sample_indices]
                y_train_boot = self.y_train.iloc[sample_indices]
            
            # Train model on bootstrap sample
            if model_type == 'random_forest':
                boot_model = RandomForestClassifier(
                    n_estimators=model.n_estimators,
                    max_depth=model.max_depth,
                    min_samples_split=model.min_samples_split,
                    min_samples_leaf=model.min_samples_leaf,
                    max_features=model.max_features,
                    random_state=self.random_state + i,
                    n_jobs=-1,
                    class_weight='balanced'
                )
            else:
                boot_model = LogisticRegression(
                    random_state=self.random_state + i,
                    max_iter=500,
                    solver='lbfgs',
                    C=model.C[0] if hasattr(model.C, '__len__') else model.C
                )
            
            boot_model.fit(X_train_boot, y_train_boot)
            bootstrap_predictions[:, i] = boot_model.predict_proba(X_data)[:, 1]
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=1)
        ci_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=1)
        ci_std = np.std(bootstrap_predictions, axis=1)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'prediction_proba': y_pred_proba,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'ci_std': ci_std,
            'actual': self.y_test.values
        })
        
        # Calculate metrics
        mean_ci_width = results_df['ci_width'].mean()
        median_ci_width = results_df['ci_width'].median()
        coverage = ((results_df['actual'] >= results_df['ci_lower']) & 
                   (results_df['actual'] <= results_df['ci_upper'])).mean()
        
        print(f"\nConfidence Interval Statistics ({confidence_level*100:.0f}% CI):")
        print(f"  Mean CI Width:   {mean_ci_width:.4f}")
        print(f"  Median CI Width: {median_ci_width:.4f}")
        print(f"  Coverage:        {coverage:.4f} ({coverage*100:.2f}%)")
        print(f"  Expected Coverage: {confidence_level:.4f} ({confidence_level*100:.0f}%)")
        
        # Show sample predictions with CIs
        print(f"\nSample Predictions with Confidence Intervals:")
        print("-" * 80)
        print(f"{'Index':<8s} | {'Pred Prob':<12s} | {'CI Lower':<10s} | {'CI Upper':<10s} | {'CI Width':<10s} | {'Actual':<6s}")
        print("-" * 80)
        for i in range(min(10, len(results_df))):
            row = results_df.iloc[i]
            print(f"{i:<8d} | {row['prediction_proba']:<12.4f} | {row['ci_lower']:<10.4f} | "
                  f"{row['ci_upper']:<10.4f} | {row['ci_width']:<10.4f} | {row['actual']:<6.0f}")
        
        return results_df
    
    def get_rf_tree_variance_intervals(self):
        """
        Calculate prediction intervals using variance across Random Forest trees.
        
        This method leverages the ensemble nature of Random Forest to estimate uncertainty.
        Each tree gives a prediction, and we use the variance across trees.
        
        Returns:
            DataFrame with predictions and variance-based intervals
        """
        if self.rf_model is None:
            print("Random Forest model not trained yet.")
            return
        
        print("\n" + "="*60)
        print("RANDOM FOREST TREE VARIANCE INTERVALS")
        print("="*60)
        
        # Get predictions from all trees
        print(f"Calculating predictions from {self.rf_model.n_estimators} trees...")
        tree_predictions = np.array([tree.predict_proba(self.X_test)[:, 1] 
                                    for tree in self.rf_model.estimators_])
        
        # Calculate statistics
        mean_pred = tree_predictions.mean(axis=0)
        std_pred = tree_predictions.std(axis=0)
        
        # Calculate confidence intervals (assuming normal distribution)
        # 95% CI: mean +/- 1.96 * std
        ci_lower = np.clip(mean_pred - 1.96 * std_pred, 0, 1)
        ci_upper = np.clip(mean_pred + 1.96 * std_pred, 0, 1)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'mean_prediction': mean_pred,
            'std_prediction': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_upper - ci_lower,
            'actual': self.y_test.values
        })
        
        # Calculate metrics
        mean_std = results_df['std_prediction'].mean()
        mean_ci_width = results_df['ci_width'].mean()
        median_ci_width = results_df['ci_width'].median()
        
        print(f"\nTree Variance Statistics:")
        print(f"  Mean Prediction Std:  {mean_std:.4f}")
        print(f"  Mean CI Width:        {mean_ci_width:.4f}")
        print(f"  Median CI Width:      {median_ci_width:.4f}")
        
        # Identify high uncertainty predictions
        high_uncertainty_threshold = results_df['std_prediction'].quantile(0.90)
        high_uncertainty = results_df[results_df['std_prediction'] > high_uncertainty_threshold]
        
        print(f"\nHigh Uncertainty Predictions (top 10%):")
        print(f"  Count: {len(high_uncertainty)}")
        print(f"  Mean Std: {high_uncertainty['std_prediction'].mean():.4f}")
        
        # Show sample predictions
        print(f"\nSample Predictions with Tree Variance Intervals:")
        print("-" * 90)
        print(f"{'Index':<8s} | {'Mean Pred':<12s} | {'Std':<10s} | {'CI Lower':<10s} | {'CI Upper':<10s} | {'Actual':<6s}")
        print("-" * 90)
        for i in range(min(10, len(results_df))):
            row = results_df.iloc[i]
            print(f"{i:<8d} | {row['mean_prediction']:<12.4f} | {row['std_prediction']:<10.4f} | "
                  f"{row['ci_lower']:<10.4f} | {row['ci_upper']:<10.4f} | {row['actual']:<6.0f}")
        
        # Visualize prediction uncertainty
        self._visualize_prediction_uncertainty(results_df)
        
        return results_df
    
    def _visualize_prediction_uncertainty(self, results_df: pd.DataFrame):
        """
        Visualize prediction uncertainty.
        
        Args:
            results_df: DataFrame with prediction statistics
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Random Forest Prediction Uncertainty Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Prediction distribution with uncertainty bands
        sorted_idx = np.argsort(results_df['mean_prediction'])
        axes[0, 0].plot(results_df['mean_prediction'].iloc[sorted_idx], 
                       label='Mean Prediction', color='blue', linewidth=2)
        axes[0, 0].fill_between(
            range(len(results_df)),
            results_df['ci_lower'].iloc[sorted_idx],
            results_df['ci_upper'].iloc[sorted_idx],
            alpha=0.3, color='blue', label='95% CI'
        )
        axes[0, 0].scatter(range(len(results_df)), 
                          results_df['actual'].iloc[sorted_idx],
                          alpha=0.2, s=5, color='red', label='Actual')
        axes[0, 0].set_title('Predictions with Confidence Intervals')
        axes[0, 0].set_xlabel('Sample (sorted by prediction)')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # 2. Uncertainty distribution
        axes[0, 1].hist(results_df['std_prediction'], bins=50, 
                       edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(results_df['std_prediction'].mean(), 
                          color='red', linestyle='--', linewidth=2, 
                          label=f"Mean: {results_df['std_prediction'].mean():.4f}")
        axes[0, 1].set_title('Distribution of Prediction Uncertainty')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # 3. Uncertainty vs Prediction
        axes[1, 0].scatter(results_df['mean_prediction'], 
                          results_df['std_prediction'],
                          alpha=0.5, s=10)
        axes[1, 0].set_title('Uncertainty vs Prediction')
        axes[1, 0].set_xlabel('Mean Prediction')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. CI Width distribution
        axes[1, 1].hist(results_df['ci_width'], bins=50,
                       edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(results_df['ci_width'].mean(),
                          color='red', linestyle='--', linewidth=2,
                          label=f"Mean: {results_df['ci_width'].mean():.4f}")
        axes[1, 1].set_title('Distribution of Confidence Interval Width')
        axes[1, 1].set_xlabel('CI Width')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rf_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n  Uncertainty visualization saved to: rf_uncertainty_analysis.png")
        plt.show()
    
    def run_complete_pipeline(self, rf_n_estimators: int = 100, rf_max_depth: int = 20,
                             rf_min_samples_split: int = 10, rf_min_samples_leaf: int = 5,
                             time_based_split: bool = False, cross_validate: bool = True):
        """
        Execute complete pipeline: train all models and compare.
        
        Args:
            rf_n_estimators: Number of trees for Random Forest
            rf_max_depth: Max depth for Random Forest (default 20 to prevent overfitting)
            rf_min_samples_split: Minimum samples required to split a node (default 10)
            rf_min_samples_leaf: Minimum samples required at a leaf node (default 5)
            time_based_split: If True, use time-based split (train on past, test on future)
            cross_validate: If True, perform k-fold cross-validation
        """
        print("\n" + "="*70)
        print(" SMART RIDE MODEL COMPARISON - COMPLETE PIPELINE")
        print("="*70)
        
        # Load and prepare data
        if time_based_split:
            X, y, pickup_datetime = self.load_and_prepare_data(return_datetime=True)
        else:
            X, y = self.load_and_prepare_data(return_datetime=False)
            pickup_datetime = None
        
        # Split data
        self.split_data(X, y, time_based=time_based_split, pickup_datetime=pickup_datetime)
        
        # Train baseline Logistic Regression
        self.train_baseline()
        
        # Train Random Forest with regularization to prevent overfitting
        self.train_random_forest(
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_split=rf_min_samples_split,
            min_samples_leaf=rf_min_samples_leaf
        )
        
        # Cross-validation (if enabled)
        cv_results = []
        if cross_validate:
            print("\n" + "="*70)
            print(" CROSS-VALIDATION ANALYSIS")
            print("="*70)
            
            if self.baseline_model is not None:
                cv_baseline = self.cross_validate_model(
                    self.baseline_model, "Baseline (Logistic Regression)", use_scaled=True
                )
                cv_results.append(cv_baseline)
            
            if self.rf_model is not None:
                cv_rf = self.cross_validate_model(
                    self.rf_model, "Random Forest", use_scaled=False
                )
                cv_results.append(cv_rf)
        
        # Compare all models
        comparison_df = self.compare_all_models()
        
        # Feature importance
        feature_importance = self.analyze_feature_importance()
        
        # Visualize comparison
        self.visualize_comparison(comparison_df, save_path='model_comparison.png')
        
        print("\n" + "="*70)
        print(" PIPELINE COMPLETED SUCCESSFULLY")
        print("="*70)
        
        return comparison_df, feature_importance


def main():
    """
    Main execution function for comprehensive model training and comparison.
    
    This trains and compares:
    - Logistic Regression (baseline)
    - Random Forest
    
    Compares their performance and provides detailed evaluation.
    """
    # Initialize and run complete pipeline
    models = SmartRideAdvancedModels(
        data_path="data/processed/uber_real_data_processed.csv"
    )
    
    comparison_df, feature_importance = models.run_complete_pipeline(
        rf_n_estimators=100,
        rf_max_depth=20,           # Limit depth to prevent overfitting
        rf_min_samples_split=10,   # Require more samples to split
        rf_min_samples_leaf=5,     # Require more samples in leaves
        time_based_split=False,    # Set to True for time-based validation (recommended for production)
        cross_validate=True        # Enable cross-validation for robust evaluation
    )
    
    # Summary
    print("\n" + "="*70)
    print(" FINAL SUMMARY")
    print("="*70)
    print("\nModel Performance Ranking (by Test Accuracy):")
    ranked = comparison_df.sort_values('test_accuracy', ascending=False)
    for idx, row in ranked.iterrows():
        print(f"  {row['model_name']:30s} | Accuracy: {row['test_accuracy']:.4f} | F1: {row['test_f1']:.4f}")
    
    print("\nBest Model:", ranked.iloc[0]['model_name'])
    print(f"  Test Accuracy: {ranked.iloc[0]['test_accuracy']:.4f}")
    print(f"  Test F1-Score: {ranked.iloc[0]['test_f1']:.4f}")
    print(f"  Test ROC-AUC: {ranked.iloc[0]['test_roc_auc']:.4f}")
    
    # SHAP Explainability Analysis
    print("\n" + "="*70)
    print(" EXPLAINABILITY & UNCERTAINTY ANALYSIS")
    print("="*70)
    
    # SHAP for Random Forest
    print("\n--- SHAP Analysis for Random Forest ---")
    shap_values_rf, shap_importance_rf = models.explain_with_shap(
        model_type='random_forest',
        sample_size=100,
        plot_type='summary',
        save_path='shap_rf_summary.png',
        show_plot=False
    )
    
    # SHAP bar plot for Random Forest
    models.explain_with_shap(
        model_type='random_forest',
        sample_size=100,
        plot_type='bar',
        save_path='shap_rf_bar.png',
        show_plot=False
    )
    
    # SHAP for Baseline (Logistic Regression)
    print("\n--- SHAP Analysis for Baseline ---")
    shap_values_baseline, shap_importance_baseline = models.explain_with_shap(
        model_type='baseline',
        sample_size=100,
        plot_type='summary',
        save_path='shap_baseline_summary.png',
        show_plot=False
    )
    
    # Single prediction explanation
    print("\n--- Single Prediction Explanation ---")
    models.explain_single_prediction(
        sample_index=0,
        model_type='random_forest',
        save_path='shap_single_prediction.png',
        show_plot=False
    )
    
    # Confidence Intervals
    print("\n" + "="*70)
    print(" CONFIDENCE INTERVAL ANALYSIS")
    print("="*70)
    
    # Tree variance-based confidence intervals (fast method)
    print("\n--- Tree Variance Intervals (Random Forest) ---")
    rf_variance_ci = models.get_rf_tree_variance_intervals()
    
    # Bootstrap confidence intervals (more robust but slower)
    print("\n--- Bootstrap Confidence Intervals (Random Forest) ---")
    rf_bootstrap_ci = models.get_prediction_confidence_intervals(
        model_type='random_forest',
        confidence_level=0.95,
        n_bootstrap=50  # Reduced for speed; increase to 100+ for production
    )
    
    print("\n--- Bootstrap Confidence Intervals (Baseline) ---")
    baseline_bootstrap_ci = models.get_prediction_confidence_intervals(
        model_type='baseline',
        confidence_level=0.95,
        n_bootstrap=50
    )
    
    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated visualizations:")
    print("  - model_comparison.png (model performance comparison)")
    print("  - shap_rf_summary.png (SHAP summary for Random Forest)")
    print("  - shap_rf_bar.png (SHAP bar chart for Random Forest)")
    print("  - shap_baseline_summary.png (SHAP summary for Baseline)")
    print("  - shap_single_prediction.png (Single prediction explanation)")
    print("  - rf_uncertainty_analysis.png (Uncertainty visualization)")
    
    return models


if __name__ == "__main__":
    main()

