# Smart Ride: Driver Decision Support

## Project Overview
This project aims to create a machine learning model that helps Uber drivers make informed decisions about whether to accept or decline ride requests by predicting ride profitability.

## Team Members
- Jeffrey Chen
- Alex Chen  
- Eric Chtilianov
- Ethan Dietrich

## Problem Statement
Uber drivers often waste time and lose money by accepting unprofitable or inefficient rides. This typically consists of long pickup times, short trip value, and a high chance of cancellation. Our model will act as a decision-support system to predict whether a ride is worth accepting, factoring in distance, time, price, and cancellation patterns.

## Dataset
- **Source**: Uber Ride Analytics Dataset 2024
- **Problem Type**: Binary Classification (Accept vs Reject)
- **Target Variable**: Ride profitability (derived from multiple factors)

## Approach
1. **Baseline Model**: Logistic Regression to establish profitability thresholds
2. **Feature Engineering**: Extract relevant features from ride data
3. **Classification**: Convert regression output to accept/reject decisions
4. **Evaluation**: Accuracy, F1, ROC-AUC, and simulated driver earnings improvement

## Project Structure
```
├── data/                        # Raw and processed datasets
├── src/                         # Source code modules
│   ├── preprocessing_data.py    # Data cleaning and preprocessing
│   ├── baseline_model.py        # ML model implementations    
└── requirements.txt             # Python dependencies
```
