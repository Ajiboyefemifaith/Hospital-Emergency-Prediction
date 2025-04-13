from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Create an empty dictionary to store the loaded models
models = {}

# Define file paths for the 5 models
model_files = {
    "SVM": os.path.join("models", "SVM_model.pkl"),
    "KNN": os.path.join("models", "KNN_model.pkl"),
    "XGBoost": os.path.join("models", "XGBoost_model.pkl"),
    "LightGBM": os.path.join("models", "LightGBM_model.pkl"),
    "CatBoost": os.path.join("models", "CatBoost_model.pkl"),
}

# Load models into the dictionary
for name, filepath in model_files.items():
    try:
        models[name] = joblib.load(filepath)
    except Exception as e:
        print(f"Error loading model {name}: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles prediction requests. 
    Expects JSON input with a "data" key containing a single record or a list of records.
    Returns model predictions in JSON format.
    """
    input_data = request.get_json()
    
    if not input_data or "data" not in input_data:
        return jsonify({"error": "Invalid input. JSON must contain 'data' key."}), 400

    sample = input_data["data"]

    try:
        df = pd.DataFrame(sample if isinstance(sample, list) else [sample])
    except Exception as e:
        return jsonify({"error": f"Invalid input data format: {e}"}), 400

    predictions = {}
    for name, model in models.items():
        try:
            preds = model.predict(df)
            predictions[name] = preds.tolist()
        except Exception as e:
            predictions[name] = f"Error: {e}"

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


#single record

{
    "data": {
        "Age": 35,
        "Gender": "Male",
        "BloodType": "O+",
        "InsuranceProvider": "HealthTrust",
        "BillingAmount": 1250.75,
        "TestResults": 98.5,
        "LengthOfStay": 4
    }
}


# Two records
{
    "data": [
        {
            "Age": 35,
            "Gender": "Male",
            "BloodType": "O+",
            "InsuranceProvider": "HealthTrust",
            "BillingAmount": 1250.75,
            "TestResults": 98.5,
            "LengthOfStay": 4
        },
        {
            "Age": 42,
            "Gender": "Female",
            "BloodType": "A-",
            "InsuranceProvider": "MediCare Plus",
            "BillingAmount": 2345.50,
            "TestResults": 89.3,
            "LengthOfStay": 7
        }
    ]
}
