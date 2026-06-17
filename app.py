import joblib
from pydantic import BaseModel
import pandas as pd

from fastapi import FastAPI

import mlflow

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
    with mlflow.start_run():
        mlflow.log_param("Churned", churned )
        mlflow.log_param("Contract", user_input.Contract )
        mlflow.log_metric("Probability", prob_churn_yes)
        mlflow.log_metric("Monthly Charges", user_input.MonthlyCharges )
        mlflow.log_metric("Total Charges", user_input.TotalCharges)
    return {"churned" :  churned,
            "probability" : float(  prob_churn_yes)
            } 




