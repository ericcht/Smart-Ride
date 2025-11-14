"""
Smart Ride Real Data Preprocessing Module
Author: Team 5 (Jeffery Chen, Alex Chen, Eric Chtilianov, Ethan Dietrich)

This module handles the actual Uber Ride Analytics dataset from Kaggle.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class SmartRideRealDataPreprocessor:
    """
    Preprocessing pipeline for the real Uber Ride Analytics dataset.
    
    This class handles:
    - Loading the actual Kaggle dataset
    - Mapping columns to our expected format
    - Creating profitability-based target variables
    - Feature engineering specific to this dataset
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the preprocessor for real data.
        
        Args:
            data_path: Path to the real dataset file
        """
        self.data_path = data_path or "data/raw/uber_data.csv"
        self.raw_data = None
        self.processed_data = None
        
        # Column mapping from real dataset to our expected format
        self.column_mapping = {
            'Booking ID': 'ride_id',
            'Customer ID': 'rider_id',
            'Date': 'pickup_date',
            'Time': 'pickup_time',
            'Vehicle Type': 'vehicle_type',
            'Pickup Location': 'pickup_location',
            'Drop Location': 'drop_location',
            'Booking Value': 'fare_amount',
            'Ride Distance': 'trip_distance',
            'Avg VTAT': 'wait_time',
            'Avg CTAT': 'trip_duration',
            'Driver Ratings': 'driver_rating',
            'Customer Rating': 'customer_rating',
            'Payment Method': 'payment_type',
            'Booking Status': 'booking_status'
        }
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the real Uber dataset.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Raw dataset as pandas DataFrame
        """
        if file_path:
            self.data_path = file_path
            
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        print(f"Loading dataset from: {self.data_path}")
        self.raw_data = pd.read_csv(self.data_path)
        
        print(f"Loaded dataset with shape: {self.raw_data.shape}")
        print(f"Columns: {list(self.raw_data.columns)}")
        
        return self.raw_data
    
    def clean_and_map_data(self) -> pd.DataFrame:
        """
        Clean the raw data and map columns to our expected format.
        
        Returns:
            Cleaned and mapped DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = self.raw_data.copy()
        
        # Remove quotes from string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip('"')
        
        # Filter for completed rides only (remove cancelled/incomplete)
        print(f"Original dataset shape: {df.shape}")
        
        # Keep completed rides and rides with booking values
        completed_mask = (
            (df['Booking Status'] == 'Completed') & 
            (df['Booking Value'].notna()) & 
            (df['Ride Distance'].notna()) &
            (df['Booking Value'] > 0) &
            (df['Ride Distance'] > 0)
        )
        
        df = df[completed_mask].copy()
        print(f"After filtering completed rides: {df.shape}")
        
        # Create datetime column
        df['pickup_datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
        
        # Remove rows with invalid datetime
        df = df.dropna(subset=['pickup_datetime'])
        print(f"After datetime cleaning: {df.shape}")
        
        # Map columns to our expected format
        mapped_df = pd.DataFrame()
        for old_col, new_col in self.column_mapping.items():
            if old_col in df.columns:
                mapped_df[new_col] = df[old_col]
        
        # Add the datetime column
        mapped_df['pickup_datetime'] = df['pickup_datetime']
        
        # Add additional columns we need
        mapped_df['tip_amount'] = 0  # Tips not available in this dataset
        mapped_df['total_amount'] = mapped_df['fare_amount']  # Same as fare for this dataset
        
        # Add driver_id (not available, so create dummy)
        mapped_df['driver_id'] = ['driver_' + str(i).zfill(4) for i in range(1, len(mapped_df) + 1)]
        
        # Handle missing values
        mapped_df['driver_rating'] = mapped_df['driver_rating'].fillna(mapped_df['driver_rating'].median())
        mapped_df['customer_rating'] = mapped_df['customer_rating'].fillna(mapped_df['customer_rating'].median())
        mapped_df['wait_time'] = mapped_df['wait_time'].fillna(mapped_df['wait_time'].median())
        mapped_df['trip_duration'] = mapped_df['trip_duration'].fillna(mapped_df['trip_duration'].median())
        
        # Time columns are already in minutes (VTAT = Vehicle Turnaround Time, CTAT = Customer Turnaround Time)
        # No conversion needed - they're already in the correct units
        
        # Add pickup time (not available, estimate from wait time)
        np.random.seed(42)
        mapped_df['pickup_time'] = mapped_df['wait_time'] * np.random.uniform(0.5, 2.0, len(mapped_df))
        
        print(f"Final cleaned dataset shape: {mapped_df.shape}")
        return mapped_df
    
    def encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using one-hot encoding.
        
        Args:
            data: DataFrame with categorical columns
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = data.copy()
        
        # Categorical columns to encode
        categorical_columns = ['payment_type', 'vehicle_type', 'booking_status']
        
        for col in categorical_columns:
            if col in df.columns:
                # One-hot encode with prefix and handle missing values
                # need x-1 columns, so we need to drop one
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
                
        return df
    
    def create_profitability_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create profitability-based features and target variable.
        
        Args:
            data: Cleaned DataFrame
            
        Returns:
            DataFrame with profitability features
        """
        df = data.copy()
        
        # Extract time features
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Calculate efficiency metrics
        df['fare_per_km'] = df['fare_amount'] / (df['trip_distance'] + 1e-6)
        df['fare_per_minute'] = df['fare_amount'] / (df['trip_duration'] + 1e-6)
        df['speed_kmh'] = df['trip_distance'] / (df['trip_duration'] / 60 + 1e-6)
        
        # Create improved profitability score
        # Factor 1: Fare efficiency per km (normalized to percentile)
        fare_efficiency = df['fare_per_km'].rank(pct=True)
        
        # Factor 2: Time efficiency per minute (NEW - earnings per minute)
        time_efficiency = df['fare_per_minute'].rank(pct=True)
        
        # Factor 3: Distance preference (categorical scoring)
        # Short (0-10 km): Good turnover, frequent rides
        # Medium (10-25 km): Optimal range - good balance of fare and time
        # Long (25-40 km): Acceptable but less ideal
        # Very Long (40+ km): Penalize - too much time commitment
        def calculate_distance_score(distance):
            """Score rides based on distance category."""
            if distance <= 10:
                return 0.8  # Short: good turnover
            elif distance <= 25:
                return 1.0  # Medium: optimal range
            elif distance <= 40:
                return 0.6  # Long: acceptable but less ideal
            else:
                return 0.3  # Very long: penalize
        
        distance_score = df['trip_distance'].apply(calculate_distance_score)
        
        # Factor 4: Enhanced time premium (hour-specific bonuses)
        hour_bonus = df['pickup_hour'].apply(lambda h: 
            0.3 if h in [7, 8, 9] else  # Morning rush
            0.3 if h in [17, 18, 19] else  # Evening rush
            0.15 if h in [20, 21, 22] else  # Late evening (surge pricing)
            0.1 if h in [10, 11, 12, 13, 14, 15, 16] else  # Daytime
            0.05  # Late night/early morning
        )
        weekend_bonus = df['is_weekend'] * 0.2
        time_premium = hour_bonus + weekend_bonus
        # Normalize to 0-1 range
        if time_premium.max() > 0:
            time_premium = time_premium / time_premium.max()
        
        # Factor 5: Rating bonus
        rating_bonus = (df['driver_rating'] - 3) / 2 * 0.1  # Scale rating to 0-0.1 bonus
        rating_bonus = np.maximum(0, rating_bonus)
        # Normalize to 0-1 range
        if rating_bonus.max() > 0:
            rating_bonus = rating_bonus / rating_bonus.max()
        
        # Factor 6: Wait time penalty (shorter is better)
        wait_penalty = 1 / (1 + df['wait_time'] / 5)  # Already 0-1 range
        
        # Factor 7: Duration efficiency (NEW - penalize very long trips)
        # Shorter trips = more turnover = better profitability
        duration_efficiency = 1 / (1 + df['trip_duration'] / 30)  # Penalize trips > 30 min
        
        # Combine factors with improved weights
        df['profitability_score'] = (
            fare_efficiency * 0.26 +     # Fare efficiency is more important
            time_efficiency * 0.21 +     # Time efficiency is more important
            distance_score * 0.20 +      # Distance is important
            time_premium * 0.15 +        # Time premium is important
            rating_bonus * 0.10 +        # Rating bonus is important
            wait_penalty * 0.05 +        # Wait penalty is important
            duration_efficiency * 0.03  # Duration efficiency is important
        )
        
        # Improved threshold: use percentile instead of median
        # Top 40% are "should accept" (more selective than 50% median)
        threshold_percentile = 0.4
        threshold_value = df['profitability_score'].quantile(1 - threshold_percentile)
        df['should_accept'] = (df['profitability_score'] > threshold_value).astype(int)
        
        print(f"Target distribution: {df['should_accept'].value_counts().to_dict()}")
        print(f"Average profitability score: {df['profitability_score'].mean():.3f}")
        print(f"Threshold (top {threshold_percentile*100}%): {threshold_value:.3f}")
        
        return df
    
    def preprocess_real_data(self, file_path: str = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline for real data.
        
        Args:
            file_path: Path to raw dataset
            
        Returns:
            Tuple of (processed_data, preprocessing_info)
        """
        print("Starting real data preprocessing pipeline...")
        
        # Load data
        self.load_data(file_path)
        
        # Clean and map data
        cleaned_data = self.clean_and_map_data()
        
        # Encode categorical features
        encoded_data = self.encode_categorical_features(cleaned_data)
        
        # Create profitability features
        processed_data = self.create_profitability_features(encoded_data)
        
        # Store processed data
        self.processed_data = processed_data
        
        # Create preprocessing info
        preprocessing_info = {
            'original_shape': self.raw_data.shape,
            'processed_shape': processed_data.shape,
            'target_column': 'should_accept',
            'target_distribution': processed_data['should_accept'].value_counts().to_dict(),
            'average_fare': processed_data['fare_amount'].mean(),
            'average_distance': processed_data['trip_distance'].mean(),
            'average_duration': processed_data['trip_duration'].mean()
        }
        
        print("Real data preprocessing pipeline completed successfully!")
        print(f"Processed data shape: {processed_data.shape}")
        
        return processed_data, preprocessing_info
    
    def save_processed_data(self, output_path: str = None):
        """
        Save the processed data to file.
        
        Args:
            output_path: Path to save processed data
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run preprocess_real_data() first.")
            
        if output_path is None:
            output_path = "data/processed/uber_real_data_processed.csv"
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.processed_data.to_csv(output_path, index=False)
        print(f"Processed real data saved to: {output_path}")


def main():
    # Initialize preprocessor
    preprocessor = SmartRideRealDataPreprocessor()
    
    # Process the real data
    processed_data, info = preprocessor.preprocess_real_data()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    print("\nPreprocessing Summary:")
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
