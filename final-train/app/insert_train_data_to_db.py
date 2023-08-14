import pandas as pd
from sqlalchemy.sql import text as sa_text
from database import engine,create_db_and_tables
from models import TrainConsumption
from sqlmodel import Session

# Read and transform data
df = pd.read_csv("GercekZamanliTuketim-15062020-15062023.csv", encoding= 'unicode_escape')
print(df.head())

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

df = df[["CONSUMPTION", "YEAR", "MONTH", "DAY", "HOUR"]]

print(df)

create_db_and_tables()

# Truncate table with sqlalchemy
with Session(engine) as session:
    session.execute(sa_text(''' TRUNCATE TABLE trainconsumption '''))
    session.commit()

# Insert training data
records_to_insert = []

for df_idx, line in df.iterrows():
    records_to_insert.append(
                    TrainConsumption(
                        CONSUMPTION = line[0],
                        YEAR=line[1],
                        MONTH=line[2],
                        DAY=line[3],
                        HOUR=line[4]
                    )
    )

session.bulk_save_objects(records_to_insert)
session.commit()
# Ends database insertion
