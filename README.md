- About project:
    This is churn prediction system. Giving specific information about telecom client it can predict whether client goes to churn or not. The project contains API , container and monitoring system so far. The prediction model uses XGBoost for training.

- How to run the program locally:
    uvicorn app:app --reload

- How to run the program with docker:
    docker build -t my_app .
    docker run -p 8000:8000 my_app

- Dataset:
    You can find Telco Dataset on Kaggle: [Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    
- MLflow Monitoring:
    MLflow Monitoring system logs
        parameters: Churned and Contract type.
    and also logs
        metrics: Probability of churn, Monthly charges and Total charges. 
