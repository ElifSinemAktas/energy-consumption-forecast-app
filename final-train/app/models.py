from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field

class TrainConsumption(SQLModel, table=True):
    ID: Optional[int] = Field(default=None, primary_key=True)
    CONSUMPTION: float
    YEAR: int
    MONTH: int
    DAY: int
    HOUR: int
    
class TrainConsumptionDynamic(SQLModel, table=True):
    ID: Optional[int] = Field(default=None, primary_key=True)
    CONSUMPTION: float
    YEAR: int
    MONTH: int
    DAY: int
    HOUR: int
