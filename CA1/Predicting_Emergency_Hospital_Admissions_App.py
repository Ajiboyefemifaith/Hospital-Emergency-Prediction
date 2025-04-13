import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data/processed_data.csv')
df_raw = pd.read_csv('data/healthcare_dataset.csv')

# Function to display data preview
def display_data():
    st.header('Data Preview')
    st.subheader('Raw Data')
    st.write(df_raw.head(10))
    st.subheader('Processed Data')
    st.write(df.head(10))

# Function to display visualizations
def display_visualizations():
    st.header('Visualizations')
    for viz in os.listdir('visualizations'):
        st.subheader(viz.replace('.png', '').replace('_', ' ').title())
        img = plt.imread(os.path.join('visualizations', viz))
        st.image(img)

# Function to display model metrics
def display_model_metrics():
    st.header('Model Metrics')
    for report in os.listdir('model_reports'):
        if 'classification_report' in report:
            st.subheader(report.replace('_classification_report.csv', '').replace('_', ' '))
            df_report = pd.read_csv(os.path.join('model_reports', report), index_col=0)
            st.table(df_report)

# Function to display confusion matrices
def display_confusion_matrices():
    st.header('Confusion Matrices')
    for cm_file in os.listdir('model_reports'):
        if 'confusion_matrix' in cm_file:
            st.subheader(cm_file.replace('_confusion_matrix.csv', '').replace('_', ' '))
            df_cm = pd.read_csv(os.path.join('model_reports', cm_file), index_col=0)
            fig, ax = plt.subplots()
            sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

# Function to display model comparisons
def display_model_comparisons():
    st.header('Model Comparisons')
    metrics = ['precision', 'recall', 'f1-score']
    comparison_df = pd.DataFrame(columns=['Model'] + metrics)
    
    for report in os.listdir('model_reports'):
        if 'classification_report' in report:
            model_name = report.replace('_classification_report.csv', '').replace('_', ' ')
            df_report = pd.read_csv(os.path.join('model_reports', report), index_col=0)
            avg_metrics = df_report.loc['weighted avg', metrics]
            avg_metrics['Model'] = model_name
            comparison_df = pd.concat([comparison_df, avg_metrics.to_frame().T], ignore_index=True)
    
    comparison_df[metrics] = comparison_df[metrics].apply(pd.to_numeric)
    st.subheader('Model Comparison Table')
    st.table(comparison_df.set_index('Model'))
    
    st.subheader('Model Comparison Plot')
    comparison_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
    st.pyplot(plt)

# Main function
def main():
    st.title('Emergency Hospital Admissions Prediction')
    menu = ['Data Preview', 'Visualizations', 'Model Metrics', 'Confusion Matrices', 'Model Comparisons']
    choice = st.sidebar.selectbox('Menu', menu)
    
    if choice == 'Data Preview':
        display_data()
    elif choice == 'Visualizations':
        display_visualizations()
    elif choice == 'Model Metrics':
        display_model_metrics()
    elif choice == 'Confusion Matrices':
        display_confusion_matrices()
    elif choice == 'Model Comparisons':
        display_model_comparisons()

if __name__ == '__main__':
    main()
