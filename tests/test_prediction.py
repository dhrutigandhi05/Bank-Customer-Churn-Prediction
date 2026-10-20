from src.predict import predict_churn

def test_prediction_output():
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
    assert result["prediction"] in ["Stayed", "Churned"]
    assert 0 <= result["churn_probability"] <= 1
    assert isinstance(result["churn_probability"], float)