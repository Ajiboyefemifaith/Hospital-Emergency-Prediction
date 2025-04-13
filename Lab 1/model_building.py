import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib
import os

# Load the preprocessed data
df = pd.read_csv('data/processed_data.csv')

# Split the data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Function to train models
def train_models(X_train, y_train):
    models = {}
    
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['LogisticRegression'] = lr
    
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf
    
    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    models['GradientBoosting'] = gb
    
    return models

# Train the models
models = train_models(X_train, y_train)

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    if not os.path.exists('model_reports'):
        os.makedirs('model_reports')
    if not os.path.exists('models'):
        os.makedirs('models')
    reports = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        reports[name] = {'report': report, 'confusion_matrix': cm}
        
        # Save the classification report
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(f'model_reports/{name}_classification_report.csv', index=True)
        
        # Save the confusion matrix
        df_cm = pd.DataFrame(cm, index=['Actual_No', 'Actual_Yes'], columns=['Predicted_No', 'Predicted_Yes'])
        df_cm.to_csv(f'model_reports/{name}_confusion_matrix.csv', index=True)
        
        # Save the model
        joblib.dump(model, f'models/{name}_model.pkl')
    
    return reports

# Evaluate the models
reports = evaluate_models(models, X_test, y_test)
