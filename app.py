import joblib
from pydantic import BaseModel
import pandas as pd

from fastapi import FastAPI

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

app = FastAPI()

class CustomerData(BaseModel):
    SeniorCitizen : int
    Partner : str
    Dependents : str
    tenure : int 
    PhoneService : str
    MultipleLines : str
    InternetService : str
    OnlineSecurity : str
    OnlineBackup : str
    DeviceProtection : str
    TechSupport : str
    StreamingTV : str
    StreamingMovies : str
    Contract : str
    PaperlessBilling : str
    PaymentMethod : str
    MonthlyCharges : float 
    TotalCharges : float 


to_binary = [ "Partner", "Dependents", "PhoneService", 
             "PaperlessBilling", "MultipleLines","OnlineSecurity", 
             "OnlineBackup", "DeviceProtection", "TechSupport",
             "StreamingTV", "StreamingMovies"]


to_hot_coding = [ "InternetService", "Contract", "PaymentMethod" ]


def pre_processing( user_input: CustomerData ):

    user_dict = user_input.model_dump()
    
    for feature in to_binary:
        current = user_dict[feature]
        user_dict[feature] = 1 if current == "Yes" else 0

    data = pd.DataFrame([user_dict])
    data = pd.get_dummies(data, dtype=int)

    data = data.reindex(columns=feature_columns, fill_value=0)
    
    data_scaled = scaler.transform(data)
    return data_scaled


@app.post("/predict")
def predict( user_input: CustomerData ):
    data_scaled = pre_processing(user_input)

    _, prob_churn_yes  = model.predict_proba(data_scaled)[0]

    churned = "No" if ( prob_churn_yes < 0.5 ) else "Yes"

    return {"churned" :  churned,
            "probability" : 100 * prob_churn_yes
            } 

customer_data = {
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 24,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "No",
    "DeviceProtection": "Yes",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.75,
    "TotalCharges": 120.5
}

user = CustomerData(**customer_data)
print(pre_processing(user))