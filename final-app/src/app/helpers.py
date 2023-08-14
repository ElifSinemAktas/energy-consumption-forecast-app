from scipy.stats import ks_2samp
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import timedelta
import pandas as pd
try:
    from app.models import DailyConsumption, HourlyConsumption
except:
    from models import DailyConsumption, HourlyConsumption

def print_models_info(mv):
    for m in mv:
        print(m)
        print(m.version)
        return int(m.version)

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

    df = pd.DataFrame(data=row_list_day, columns=["YEAR", "MONTH", "DAY"])
    # Predict
    prediction = model.predict(df)
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

    df = pd.DataFrame(data=row_list_hour, columns=["YEAR", "MONTH", "DAY", "HOUR"])

    # Predict
    prediction = model.predict(df)
    new_df = df.assign(CONSUMPTION=prediction)
    print(new_df)
    print(prediction)
    print(type(prediction))
    return prediction, new_df


def detect_drift(data1, data2):
    ks_result = ks_2samp(data1, data2)
    if ks_result.pvalue < 0.05:
        return "Drift exits"
    else:
        return "No drift"
    
def create_message(data1, data2, list_of_cols):
    dicty = {}
    for col in list_of_cols:
        msg = detect_drift(data1[col], data2[col])
        dicty[col] = msg
    return dicty

def html_template(hourly_results, daily_results):
    # HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <title>Weekly Drift Results</title>
    </head>
    <body>
      <h1>Results for hour-based forecast:</h1>
      <ul>
        {hourly_results}
      </ul>

      <h1>Results for day-based forecast:</h1>
      <ul>
        {daily_results}
      </ul>
    </body>
    </html>
    """

    # Format the HTML template with the dynamic data
    hourly_messages = ""
    for key, value in hourly_results.items():
        message = f"<li>{key}: {value}</li>"
        hourly_messages += message

    daily_messages = ""
    for key, value in daily_results.items():
        message = f"<li>{key}: {value}</li>"
        daily_messages += message

    formatted_html = html_template.format(hourly_results=hourly_messages, daily_results=daily_messages)
    
    return formatted_html

def mail_send(subject: str, formatted_html: str, sender_email: str , smtp_password: str, smtp_port: int, smtp_server: str, smtp_username: str, receiver_email: str):
    # Create a multipart message and set the headers
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    # Attach the HTML content to the email
    message.attach(MIMEText(formatted_html, "html"))
    
    try:
        # Establish a secure connection
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()

        # Log in to the email account
        server.login(smtp_username, smtp_password)

        # Send the email
        server.sendmail(sender_email, receiver_email, message.as_string())

        print("Email sent successfully!")
    except smtplib.SMTPException as e:
        print("Error occurred while sending the email:", str(e))
    finally:
        # Close the connection
        server.quit()