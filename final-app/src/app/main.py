from fastapi import FastAPI, Depends, Request, status
import os
from sqlmodel import Session
from mlflow.sklearn import load_model
from mlflow.tracking import MlflowClient

import pandas as pd
from datetime import datetime
try:
    from app.models import ConsumptionDriftInput
    from app.database import engine, get_db, create_db_and_tables
    from app.helpers import *
    from app.conn import *
except:
    from models import ConsumptionDriftInput
    from database import engine, get_db, create_db_and_tables
    from helpers import *
    from conn import *

# Tell where is the tracking server and artifact server
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio:9000/"

registered_model_name_1 = "EpiasRandomForestHourly"
registered_model_name_2 = "EpiasRandomForestDaily"
client = MlflowClient()

# Learn, decide and get model from mlflow model registry
model_name_daily = "EpiasRandomForestDaily"
model_version_daily = print_models_info(client.get_latest_versions(registered_model_name_2, stages=["None"]))
model_daily = load_model(model_uri=f"models:/{model_name_daily}/{model_version_daily}")

model_name_hourly = "EpiasRandomForestHourly"
model_version_hourly = print_models_info(client.get_latest_versions(registered_model_name_1, stages=["None"]))
model_hourly = load_model(
    model_uri=f"models:/{model_name_hourly}/{model_version_hourly}"
)

app = FastAPI()

# Creates all the tables defined in models module
create_db_and_tables()

# Daily Prediction endpoint
@app.post("/prediction/daily/{date}/{days}", status_code=status.HTTP_200_OK)
def predict_daily(
    date: str, days: int, fastapi_req: Request, db: Session = Depends(get_db)
):
    try:
        date_converted = datetime.strptime(date, "%Y-%m-%d")
        print(date_converted)
    except:
        return {
            "Invalid input. Please write in this format --> date: 2023-06-19(year-month-day)"
        }

    prediction, new_df = make_daily_consump_pred(
        datetime=date_converted, days=days, model=model_daily
    )
    insert_daily_consump(df=new_df, client_ip=fastapi_req.client.host, db=db)
    return {"Message": "Predictions saved to db", "Results": prediction.tolist()}


# Hourly Prediction endpoint
@app.post("/prediction/hourly/{date}/{hour}/{hours}", status_code=status.HTTP_200_OK)
def predict_hourly(
    date: str,
    hour: str,
    hours: int,
    fastapi_req: Request,
    db: Session = Depends(get_db),
):
    try:
        date_and_hour = date + " " + hour
        date_converted = datetime.strptime(date_and_hour, "%Y-%m-%d %H")
    except:
        return {
            "Invalid input. Please write in this format --> date: 2023-06-19 (year-month-day), hour: 10"
        }

    prediction, new_df = make_hourly_consump_pred(
        datetime=date_converted, hours=hours, model=model_hourly
    )
    insert_hourly_consump(df=new_df, client_ip=fastapi_req.client.host, db=db)
    return {"Message": "Predictions saved to db", "Results": prediction.tolist()}


@app.post("/drift/consumption")
async def detect(request: ConsumptionDriftInput):
    train_df = pd.read_sql("select * from trainconsumption", engine)
    
    prediction_hourly_df = pd.read_sql(f"""select * from hourlyconsumption 
                                            where PREDICTION_TIME >
                                            current_date - {request.n_days_before}""", engine)

    prediction_daily_df = pd.read_sql(f"""select * from dailyconsumption 
                                            where PREDICTION_TIME >
                                            current_date - {request.n_days_before}""", engine)

    drift_msg_hourly = create_message(train_df, prediction_hourly_df, ["YEAR", "MONTH", "DAY", "HOUR","CONSUMPTION"])
    drift_msg_daily = create_message(train_df, prediction_daily_df, ["YEAR", "MONTH", "DAY","CONSUMPTION"])

    print(drift_msg_hourly)
    print(drift_msg_daily)

    formatted_html = html_template(drift_msg_hourly,drift_msg_daily)

    mail_send(subject=subject, 
          formatted_html=formatted_html, 
          sender_email=sender_email , 
          smtp_password=smtp_app_password,
          smtp_username=smtp_username,
          smtp_port=smtp_port, 
          smtp_server=smtp_server, 
          receiver_email=receiver_email)

    return {"Message": "Drift results has also sent as email","Drift for hour-based data": drift_msg_hourly, "Drift for day-based data": drift_msg_daily }

# Welcome page
@app.get("/")
async def root():
    return {"data": "Welcome to MLOps FINAL"}
