from fastapi import FastAPI, Depends, Request, status
import os
from app.sqlmodels import DailyConsumption, HourlyConsumption
from app.database import engine, get_db, create_db_and_tables
from sqlalchemy.orm import Session
from mlflow.sklearn import load_model
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Tell where is the tracking server and artifact server
os.environ["MLFLOW_TRACKING_URI"] = "http://192.168.1.41:5000/"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://192.168.1.41:9000/"

# Learn, decide and get model from mlflow model registry
model_name_daily = "EpiasXGBoostDaily"
model_version_daily = 1
model_daily = load_model(model_uri=f"models:/{model_name_daily}/{model_version_daily}")

model_name_hourly = "EpiasXGBoostHourly"
model_version_hourly = 1
model_hourly = load_model(
    model_uri=f"models:/{model_name_hourly}/{model_version_hourly}"
)

app = FastAPI()

# Creates all the tables defined in models module
create_db_and_tables()


def insert_daily_consump(df, client_ip, db):
    for index, row in df.iterrows():
        new_pred = DailyConsumption(
            YEAR=row["YEAR"][0],
            MONTH=row["MONTH"][0],
            DAY=row["DAY"],
            CONSUMPTION=row["CONSUMPTION"],
            CLIENT_IP=client_ip,
        )

        with db as session:
            session.add(new_pred)
            session.commit()
            session.refresh(new_pred)
    return df


def insert_hourly_consump(df, client_ip, db):
    for index, row in df.iterrows():
        new_pred = HourlyConsumption(
            YEAR=row["YEAR"][0],
            MONTH=row["MONTH"][0],
            DAY=row["DAY"],
            HOUR=row["HOUR"],
            CONSUMPTION=row["CONSUMPTION"],
            CLIENT_IP=client_ip,
        )

        with db as session:
            session.add(new_pred)
            session.commit()
            session.refresh(new_pred)
    return df


# prediction functions
def make_daily_consump_pred(model, datetime, days):
    # Create list for input
    row_list_day = []
    for i in range(days):
        print(i)
        datetime_delta = datetime + timedelta(days=i + 1)
        YEAR = (datetime_delta.year,)
        MONTH = (datetime_delta.month,)
        DAY = datetime_delta.day

        row = [YEAR, MONTH, DAY]
        row_list_day.append(row)
    print(row_list_day)
    print(len(row_list_day))

    df = pd.DataFrame(data=row_list_day, columns=model.feature_names_in_)
    # Predict
    prediction = model_daily.predict(df)
    new_df = df.assign(CONSUMPTION=prediction)
    print(new_df)
    print(prediction)
    print(type(prediction))

    return prediction, new_df


def make_hourly_consump_pred(model, datetime, hours):
    # Create list for input
    row_list_hour = []
    for i in range(hours):
        print(i)
        datetime_delta = datetime + timedelta(hours=i + 1)
        YEAR = (datetime_delta.year,)
        MONTH = (datetime_delta.month,)
        DAY = datetime_delta.day
        HOUR = datetime_delta.hour
        row = [YEAR, MONTH, DAY, HOUR]
        row_list_hour.append(row)
    print(row_list_hour)
    print(len(row_list_hour))

    df = pd.DataFrame(data=row_list_hour, columns=model.feature_names_in_)

    # Predict
    prediction = model.predict(df)
    new_df = df.assign(CONSUMPTION=prediction)
    print(new_df)
    print(prediction)
    print(type(prediction))
    return prediction, new_df


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

# # Test 
# @app.post("/prediction/daily_test/{date}/{days}", status_code=status.HTTP_200_OK)
# def predict_daily(
#     date: str, days: int, fastapi_req: Request, db: Session = Depends(get_db)
# ):
#     try:
#         date_converted = datetime.strptime(date, "%Y-%m-%d")
#         print(date_converted)
#     except:
#         return {
#             "Invalid input. Please write in this format --> date: 2023-06-19(year-month-day)"
#         }

#     prediction, new_df = make_daily_consump_pred(
#         datetime=date_converted, days=days, model=model_daily
#     )
#     insert_daily_consump(df=new_df, client_ip=fastapi_req.client.host, db=db)
#     return {"Message": "Predictions saved to db", "Results": prediction.tolist()}


# Welcome page
@app.get("/")
async def root():
    return {"data": "Welcome to MLOps FINAL"}
