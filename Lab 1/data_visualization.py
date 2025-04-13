import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the preprocessed data
df = pd.read_csv('data/processed_data.csv')

# Function to create visualizations
def create_visualizations(df):
    # Churn count plot
    plt.figure(figsize=(8,6))
    sns.countplot(x='Churn', data=df)
    plt.title('Churn Count')
    plt.savefig('visualizations/churn_count.png')
    plt.close()
    
    # Distribution of Monthly Charges
    plt.figure(figsize=(8,6))
    sns.histplot(df['MonthlyCharges'], kde=True, bins=30)
    plt.title('Distribution of Monthly Charges')
    plt.savefig('visualizations/monthly_charges_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12,10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()
    
    # Churn vs Tenure
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Churn', y='tenure', data=df)
    plt.title('Churn vs Tenure')
    plt.savefig('visualizations/churn_vs_tenure.png')
    plt.close()
    
    # Churn vs Monthly Charges
    plt.figure(figsize=(8,6))
    sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
    plt.title('Churn vs Monthly Charges')
    plt.savefig('visualizations/churn_vs_monthly_charges.png')
    plt.close()

# Create visualizations directory if it doesn't exist
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Generate visualizations
create_visualizations(df)
