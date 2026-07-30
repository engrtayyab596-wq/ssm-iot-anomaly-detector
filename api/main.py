from fastapi import FastAPI, BackgroundTasks, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import os

from api.database import get_db, engine
from api import models as db_models

db_models.Base.metadata.create_all(bind=engine)
app = FastAPI(title= 'IOT Anomaly Detector')

MODEL_PATH = 'models'
ssm_models = None
thresholds = None
overall_threshold = None
scaler = None
corr_weights = None
feature_columns = None


def load_models():
    global ssm_models, thresholds, overall_threshold
    global scaler, corr_weights, feature_columns
    if ssm_models is None:
        ssm_models = joblib.load(
            f'{MODEL_PATH}/ssm_models.pkl'
        )
        thresholds = joblib.load(
            f'{MODEL_PATH}/thresholds.pkl'
        )
        overall_threshold = joblib.load(
            f'{MODEL_PATH}/overall_threshold.pkl'
        )
        scaler = joblib.load(
            f'{MODEL_PATH}/scaler.pkl'
        )
        corr_weights = joblib.load(
            f'{MODEL_PATH}/corr_weights.pkl'
        )
        feature_columns = joblib.load(
            f'{MODEL_PATH}/feature_columns.pkl'
        )


class SensorReading(BaseModel):
    engine_id: int
    cycle: int
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


def compute_anomaly(data: dict, engine_id: int,
                    cycle: int, db: Session):
    load_models()

    # data is already scaled 0-1 — no need to transform
    sensor_scores = []
    for sensor in feature_columns:
        fitted_mean = (
            ssm_models[sensor].fittedvalues.mean()
        )
        actual_value = data[sensor]
        error = abs(actual_value - fitted_mean)
        weight = corr_weights[sensor]
        sensor_scores.append(error * weight)

    anomaly_score = float(np.mean(sensor_scores))
    is_anomaly = anomaly_score > overall_threshold

    reading = db_models.SensorReading(
        engine_id=engine_id,
        cycle=cycle,
        sensor_2=data['sensor_2'],
        sensor_3=data['sensor_3'],
        sensor_4=data['sensor_4'],
        sensor_7=data['sensor_7'],
        sensor_8=data['sensor_8'],
        sensor_9=data['sensor_9'],
        sensor_11=data['sensor_11'],
        sensor_12=data['sensor_12'],
        sensor_13=data['sensor_13'],
        sensor_14=data['sensor_14'],
        sensor_15=data['sensor_15'],
        sensor_17=data['sensor_17'],
        sensor_20=data['sensor_20'],
        sensor_21=data['sensor_21'],
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly
    )
    db.add(reading)
    db.commit()

    if is_anomaly:
        alert = db_models.AnomalyAlert(
            engine_id=engine_id,
            cycle=cycle,
            anomaly_score=anomaly_score,
            threshold=float(overall_threshold)
        )
        db.add(alert)
        db.commit()


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model': 'SSM Anomaly Detector',
        'description': 'IoT Engine Anomaly Detection API'
    }


@app.post('/sensor-reading')
def receive_reading(
    data: SensorReading,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    sensor_data = {
        'sensor_2': data.sensor_2,
        'sensor_3': data.sensor_3,
        'sensor_4': data.sensor_4,
        'sensor_7': data.sensor_7,
        'sensor_8': data.sensor_8,
        'sensor_9': data.sensor_9,
        'sensor_11': data.sensor_11,
        'sensor_12': data.sensor_12,
        'sensor_13': data.sensor_13,
        'sensor_14': data.sensor_14,
        'sensor_15': data.sensor_15,
        'sensor_17': data.sensor_17,
        'sensor_20': data.sensor_20,
        'sensor_21': data.sensor_21
    }

    background_tasks.add_task(
        compute_anomaly,
        sensor_data,
        data.engine_id,
        data.cycle,
        db
    )

    return {
        'status': 'received',
        'engine_id': data.engine_id,
        'cycle': data.cycle,
        'message': 'Processing anomaly detection in background'
    }


@app.get('/anomalies')
def get_anomalies(
    engine_id: Optional[int] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(db_models.AnomalyAlert)
    if engine_id:
        query = query.filter(
            db_models.AnomalyAlert.engine_id == engine_id
        )
    alerts = query.order_by(
        db_models.AnomalyAlert.created_at.desc()
    ).limit(limit).all()

    return {
        'total_alerts': len(alerts),
        'alerts': [
            {
                'engine_id': a.engine_id,
                'cycle': a.cycle,
                'anomaly_score': a.anomaly_score,
                'threshold': a.threshold,
                'created_at': str(a.created_at)
            }
            for a in alerts
        ]
    }


@app.get('/readings/{engine_id}')
def get_readings(
    engine_id: int,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    readings = db.query(db_models.SensorReading).filter(
        db_models.SensorReading.engine_id == engine_id
    ).order_by(
        db_models.SensorReading.cycle
    ).limit(limit).all()

    return {
        'engine_id': engine_id,
        'total_readings': len(readings),
        'readings': [
            {
                'cycle': r.cycle,
                'anomaly_score': r.anomaly_score,
                'is_anomaly': r.is_anomaly
            }
            for r in readings
        ]
    }
