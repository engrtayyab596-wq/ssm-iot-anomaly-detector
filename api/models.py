from sqlalchemy import Column, Integer, Float, Boolean, DateTime
from sqlalchemy.sql import func
from api.database import Base


class SensorReading(Base):
    __tablename__ = 'sensor_readings'

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(Integer)
    cycle = Column(Integer)
    setting_1 = Column(Float)
    setting_2 = Column(Float)
    setting_3 = Column(Float)
    sensor_1 = Column(Float)
    sensor_2 = Column(Float)
    sensor_3 = Column(Float)
    sensor_4 = Column(Float)
    sensor_5 = Column(Float)
    sensor_6 = Column(Float)
    sensor_7 = Column(Float)
    sensor_8 = Column(Float)
    sensor_9 = Column(Float)
    sensor_10 = Column(Float)
    sensor_11 = Column(Float)
    sensor_12 = Column(Float)
    sensor_13 = Column(Float)
    sensor_14 = Column(Float)
    sensor_15 = Column(Float)
    sensor_16 = Column(Float)
    sensor_17 = Column(Float)
    sensor_18 = Column(Float)
    sensor_19 = Column(Float)
    sensor_20 = Column(Float)
    sensor_21 = Column(Float)
    anomaly_score = Column(Float, nullable=True)
    is_anomaly = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())


class AnomalyAlert(Base):
    __tablename__ = 'anomaly_alerts'

    id = Column(Integer, primary_key=True, index=True)
    engine_id = Column(Integer)
    cycle = Column(Integer)
    anomaly_score = Column(Float)
    threshold = Column(Float)
    created_at = Column(DateTime, server_default=func.now())
