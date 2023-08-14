import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow.sklearn 
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

df = pd.read_csv("GercekZamanliTuketim-15062020-15062023.csv", encoding= 'unicode_escape')

##### DF PART ######

df.head()
df.shape

columns = ["DATE", "TIME", "CONSUMPTION"]
df.set_axis(columns, axis='columns', inplace=True)

df['DATE'] = pd.to_datetime(df['DATE'].str.replace('.','-') + ' ' +df['TIME'])
df.drop('TIME', axis=1, inplace=True)
df['CONSUMPTION'] = df['CONSUMPTION'].str.replace(',','')
df["DAY"] = df["DATE"].dt.day
df["MONTH"] = df["DATE"].dt.month
df["YEAR"] = df["DATE"].dt.year
df["HOUR"] = df["DATE"].dt.hour
df["CONSUMPTION"] = df["CONSUMPTION"].astype(float)

df_day = df.groupby(['DAY', 'MONTH', 'YEAR']).agg({'CONSUMPTION': 'sum'})

df_day = df_day.reset_index()

df_day = df_day[["CONSUMPTION", "YEAR", "MONTH", "DAY"]]
df_day[:5]

df_hour = df[["CONSUMPTION", "YEAR", "MONTH", "DAY", "HOUR"]]
df_hour[:5]


# X-y FOR HOURLY CONSUMPTION
X_hour = df_hour.iloc[:, 1:]
print(X_hour.shape)
print(type(X_hour))
print(X_hour[:3])

y_hour = df_hour.iloc[:, 0]
print(y_hour.shape)
print(type(y_hour))
print(y_hour[:3])

# X-y FOR DAILY CONSUMPTION
X_day = df_day.iloc[:, 1:]
print(X_day.shape)
print(type(X_day))
print(X_day[:3])

y_day = df_day.iloc[:, 0]
print(y_day.shape)
print(type(y_day))
print(y_day[:3])

# SPLIT DF's
X_train_hour, X_test_hour, y_train_hour, y_test_hour = train_test_split(X_hour,y_hour, test_size=0.2, random_state=42)
X_train_day, X_test_day, y_train_day, y_test_day = train_test_split(X_day,y_day, test_size=0.2, random_state=42)

##### MLFLOW PART ######

os.environ['MLFLOW_TRACKING_URI'] = 'http://192.168.1.41:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://192.168.1.41:9000/'

experiment_list = ["EpiasHour", "EpiasDay"]

for exp in experiment_list:
    if mlflow.get_experiment_by_name(exp):
        pass
    else:
        mlflow.set_experiment(exp)
        
client = MlflowClient()
exp_id_hour = client.get_experiment_by_name("EpiasHour")._experiment_id
exp_id_day = client.get_experiment_by_name("EpiasDay")._experiment_id

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

with mlflow.start_run(run_name="Random_Hourly_Consump", experiment_id=exp_id_hour) as run:
    pipeline_hour = Pipeline([
        ('ct-ohe', ColumnTransformer([('ct', OneHotEncoder(handle_unknown='ignore', categories='auto'), [0, 1, 2, 3])], remainder='passthrough')),
        ('scaler', StandardScaler(with_mean=False)),
        ('estimator', TransformedTargetRegressor(regressor=RandomForestRegressor(), transformer=StandardScaler()))
    ])

    # Fit the pipeline
    pipeline_hour.fit(X_train_hour, y_train_hour)
    y_pred_hour = pipeline_hour.predict(X_test_hour)
    print(y_pred_hour[:10])

    (rmse, mae, r2) = eval_metrics(y_test_hour, y_pred_hour)

    # mlflow.log_param("")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    registered_model_name_2 = "EpiasRandomForestHourly"
    # Model registry does not work with file store
    if tracking_url_type_store != "file" :
        mlflow.sklearn.log_model(pipeline_hour, "model")
        mlflow.sklearn.log_model(pipeline_hour, "model",registered_model_name=registered_model_name_2)
    else:
        mlflow.sklearn.log_model(pipeline_hour, "model")

        
with mlflow.start_run(run_name="Random_Daily_Consump", experiment_id=exp_id_day) as run:
    pipeline_day = Pipeline([
        ('ct-ohe', ColumnTransformer([('ct', OneHotEncoder(handle_unknown='ignore', categories='auto'), [0, 1, 2])], remainder='passthrough')),
        ('scaler', StandardScaler(with_mean=False)),
        ('estimator', TransformedTargetRegressor(regressor=RandomForestRegressor(), transformer=StandardScaler()))
    ])

    # Fit the pipeline
    pipeline_day.fit(X_train_day, y_train_day)
    y_pred_day = pipeline_day.predict(X_test_day)
    print(y_pred_day[:10])

    (rmse, mae, r2) = eval_metrics(y_test_day, y_pred_day)

    # mlflow.log_param("")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    registered_model_name_2 = "EpiasRandomForestDaily"

    # Model registry does not work with file store
    if tracking_url_type_store != "file" :
        mlflow.sklearn.log_model(pipeline_day, "model")
        mlflow.sklearn.log_model(pipeline_day, "model",registered_model_name=registered_model_name_2)
    else:
        mlflow.sklearn.log_model(pipeline_day, "model")