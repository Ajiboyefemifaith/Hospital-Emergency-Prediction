import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Function to download dataset from Kaggle
def download_kaggle_dataset():
    kaggle_dataset = 'blastchar/telco-customer-churn'  # Dataset identifier
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    os.environ['KAGGLE_USERNAME'] = 'YOUR_KAGGLE_USERNAME'  # Replace with your Kaggle username
    os.environ['KAGGLE_KEY'] = 'YOUR_KAGGLE_KEY'  # Replace with your Kaggle API key
    subprocess.call([
        'kaggle', 'datasets', 'download', kaggle_dataset, '--unzip',
        '-p', data_dir
    ])

# Download the dataset if not already downloaded
data_file = os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
if not os.path.exists(data_file):
    download_kaggle_dataset()

# Load the dataset
df_raw = pd.read_csv(data_file)  # Original DataFrame

# Convert 'TotalCharges' to numeric
df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')

# Handle missing values in 'TotalCharges'
df_raw['TotalCharges'].fillna(df_raw['TotalCharges'].median(), inplace=True)

# Data preprocessing
def preprocess_data(df):
    # Create a copy for preprocessing
    df_processed = df.copy()
    
    # Encode binary categorical variables
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1})
    
    # Encode 'SeniorCitizen' as binary (already 0 or 1)
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].astype(int)
    
    # Drop customerID as it's not needed
    df_processed.drop('customerID', axis=1, inplace=True)
    
    # Encode remaining categorical variables using dummy variables
    df_processed = pd.get_dummies(df_processed, drop_first=True, dtype= int)
    
    # Feature scaling
    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    return df_processed

# Preprocess the data
df_processed = preprocess_data(df_raw)

# Save the preprocessed data
df_processed.to_csv(os.path.join('data', 'processed_data.csv'), index=False)
