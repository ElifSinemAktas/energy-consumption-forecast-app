import pandas as pd
from sqlalchemy.sql import text as sa_text
from database import engine,create_db_and_tables
from models import TrainConsumptionDynamic
from sqlmodel import Session

# Read and transform data
df = pd.read_csv("GercekZamanliTuketim", encoding= 'unicode_escape')
print(df.head())
df = df[["CONSUMPTION", "YEAR", "MONTH", "DAY", "HOUR"]]

create_db_and_tables()

# Truncate table with sqlalchemy
with Session(engine) as session:
    session.execute(sa_text(''' TRUNCATE TABLE trainconsumptiondynamic '''))
    session.commit()

# Insert training data
records_to_insert = []

for df_idx, line in df.iterrows():
    records_to_insert.append(
                    TrainConsumptionDynamic(
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
