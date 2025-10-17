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
        
        # Create dummy coordinates (since we don't have actual lat/lng)
        np.random.seed(42)  # For reproducibility
        n_rows = len(mapped_df)
        mapped_df['pickup_latitude'] = np.random.normal(28.6139, 0.1, n_rows)  # Delhi area
        mapped_df['pickup_longitude'] = np.random.normal(77.2090, 0.1, n_rows)
        mapped_df['dropoff_latitude'] = np.random.normal(28.6139, 0.1, n_rows)
        mapped_df['dropoff_longitude'] = np.random.normal(77.2090, 0.1, n_rows)
        
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
        
        categorical_columns = ['payment_type', 'vehicle_type', 'booking_status']
        
        for col in categorical_columns:
            if col in df.columns:
                # One-hot encode with prefix and handle missing values
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
        
        # Extract temporal features
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_day_of_week'] = df['pickup_datetime'].dt.dayofweek
        df['pickup_month'] = df['pickup_datetime'].dt.month
        df['is_weekend'] = df['pickup_day_of_week'].isin([5, 6]).astype(int)
        df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        
        # Calculate efficiency metrics
        df['fare_per_km'] = df['fare_amount'] / (df['trip_distance'] + 1e-6)
        df['fare_per_minute'] = df['fare_amount'] / (df['trip_duration'] + 1e-6)
        df['speed_kmh'] = df['trip_distance'] / (df['trip_duration'] / 60 + 1e-6)
        
        # Create profitability score (weighted combination of factors)
        fare_efficiency = df['fare_per_km'].rank(pct=True)
        
        optimal_distance = 10
        distance_score = 1 - np.abs(df['trip_distance'] - optimal_distance) / optimal_distance
        distance_score = np.maximum(0, distance_score)
        
        time_premium = df['is_rush_hour'] * 0.3 + df['is_weekend'] * 0.2
        
        rating_bonus = (df['driver_rating'] - 3) / 2 * 0.1
        rating_bonus = np.maximum(0, rating_bonus)
        
        wait_penalty = 1 / (1 + df['wait_time'] / 5)
        
        df['profitability_score'] = (
            fare_efficiency * 0.3 +
            distance_score * 0.25 +
            time_premium * 0.2 +
            rating_bonus * 0.15 +
            wait_penalty * 0.1
        )
        
        # Create binary target: accept if profitability > median
        median_profitability = df['profitability_score'].median()
        df['should_accept'] = (df['profitability_score'] > median_profitability).astype(int)
        
        print(f"Target distribution: {df['should_accept'].value_counts().to_dict()}")
        print(f"Average profitability score: {df['profitability_score'].mean():.3f}")
        
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
