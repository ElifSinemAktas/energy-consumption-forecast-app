from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


class HourlyConsumption(SQLModel, table=True):
    ID: Optional[int] = Field(default=None, primary_key=True)
    YEAR: int
    MONTH: int
    DAY: int
    HOUR: int
    CONSUMPTION: float
    PREDICTION_TIME: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    CLIENT_IP: str

class DailyConsumption(SQLModel, table=True):
    ID: Optional[int] = Field(default=None, primary_key=True)
    YEAR: int
    MONTH: int
    DAY: int
    CONSUMPTION: float
    PREDICTION_TIME: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    CLIENT_IP: str

class TrainConsumption(SQLModel, table=True):
    ID: Optional[int] = Field(default=None, primary_key=True)
    CONSUMPTION: float
    YEAR: int
    MONTH: int
    DAY: int
    HOUR: int

class ConsumptionDriftInput(SQLModel):
    n_days_before: int
    e_mail: str

    class Config:
        schema_extra = {
            "example": {
                "n_days_before": 5,
                "e_mail": "elifsinem.test@gmail.com"
            }
        }
