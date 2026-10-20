from pathlib import Path
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"

def predict_churn(customer_data):
    model = joblib.load(MODEL_PATH)
    customer_df = pd.DataFrame([customer_data])
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0][1]

    return {
        "prediction": "Churned" if prediction == 1 else "Stayed",
        "churn_probability": round(float(probability), 4)
    }

if __name__ == "__main__":
    customer = {
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 5,
        "Balance": 100000,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 75000
    }

    result = predict_churn(customer)
    print("Prediction:", result["prediction"])
    print("Estimated Churn Probability:", result["churn_probability"])