# IoT Engine Anomaly Detector

A production-grade real-time anomaly detection system for industrial IoT
sensors using State Space Models (SSM), served via a FastAPI REST API with
background task processing and PostgreSQL database storage.

---

## Project Overview

This project monitors NASA turbofan engine sensor data in real time and
detects anomalies indicating engine degradation before failure occurs.
The system trains 14 independent State Space Models on healthy engine
behaviour, then flags readings that deviate significantly from learned
normal patterns.

The project introduces two advanced engineering skills not covered in
previous projects — FastAPI Background Tasks for asynchronous IoT data
processing, and Docker Compose for orchestrating multiple services
(API + PostgreSQL database) together.

---

## Results

| Metric | Score |
|---|---|
| Training detection rate (mean) | 97.97% |
| Training detection rate (min) | 80.65% |
| Engines with 100% detection | 76 / 100 |
| Training false alarm rate | 21.30% |
| Test high risk anomaly rate | 26.41% |
| Test low risk anomaly rate | 14.23% |
| Risk separation ratio | 1.86x |
| Anomaly threshold | 0.0658 (95th percentile) |

---

## Tech Stack

| Layer | Tools |
|---|---|
| Machine Learning | statsmodels, State Space Models (SSM) |
| Data Processing | pandas, numpy, scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI, uvicorn, pydantic |
| Background Tasks | FastAPI BackgroundTasks |
| Database | PostgreSQL (Docker) / SQLite (local dev) |
| ORM | SQLAlchemy |
| Testing | pytest |
| Containerisation | Docker, Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
ssm-iot-anomaly-detector/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── notebooks/
│   └── 01_eda_and_ssm.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── api/
│   ├── main.py
│   ├── database.py
│   └── models.py
├── models/
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Why State Space Models?

State Space Models are the ideal choice for IoT anomaly detection because
they model the hidden internal state of a dynamic system over time — exactly
how a turbofan engine operates. The local level SSM tracks the underlying
true sensor level, filtering out observation noise. When a sensor reading
deviates significantly from the predicted level, it signals a change in the
engine's internal state — i.e. degradation.

Unlike threshold-based alerting (flag when sensor exceeds fixed value), SSM
learns what normal looks like for each sensor specifically, making it robust
to manufacturing variation between engines.

---

## Why Background Tasks?

In real IoT systems engines send sensor readings continuously — hundreds per
minute. Synchronous processing would make the sensor wait for anomaly
detection to complete before sending the next reading. FastAPI Background
Tasks solve this by returning "received" instantly while processing happens
asynchronously — the sensor never waits.

---

## Why Docker Compose?

The system requires two services running together — the FastAPI application
and a PostgreSQL database. Docker Compose orchestrates both with a single
command. The API waits for the database health check to pass before starting,
ensuring reliable startup ordering.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/engrtayyab596-wq/ssm-iot-anomaly-detector
cd ssm-iot-anomaly-detector
```

### 2. Create conda environment

```bash
conda create -n ML_P python=3.11
conda activate ML_P
pip install -r requirements.txt
```

### 3. Add the dataset

Download the NASA CMAPSS dataset from Kaggle and place these files in data/:

```
data/train_FD001.txt
data/test_FD001.txt
data/RUL_FD001.txt
```

### 4. Generate the models

Open and run all cells in notebooks/01_eda_and_ssm.ipynb.
This trains 14 SSM models and saves them to models/.

### 5. Run locally with SQLite

```bash
uvicorn api.main:app --reload
```

### 6. Open interactive docs

```
http://127.0.0.1:8000/docs
```

---

## How to Run with Docker Compose

```bash
docker compose up --build
```

This starts both the FastAPI API and PostgreSQL database together.
The API automatically connects to PostgreSQL via the DATABASE_URL
environment variable.

---

## How to Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

### GET /health

Returns API status.

```json
{
  "status": "ok",
  "model": "SSM Anomaly Detector",
  "description": "IoT Engine Anomaly Detection API"
}
```

### POST /sensor-reading

Accepts sensor reading, returns instantly, processes in background.

Example request:

```json
{
  "engine_id": 1,
  "cycle": 156,
  "sensor_2": 0.44,
  "sensor_3": 0.39,
  "sensor_4": 0.41,
  "sensor_7": 0.58,
  "sensor_8": 0.27,
  "sensor_9": 0.17,
  "sensor_11": 0.71,
  "sensor_12": 0.62,
  "sensor_13": 0.29,
  "sensor_14": 0.21,
  "sensor_15": 0.41,
  "sensor_17": 0.40,
  "sensor_20": 0.56,
  "sensor_21": 0.58
}
```

Example response (immediate):

```json
{
  "status": "received",
  "engine_id": 1,
  "cycle": 156,
  "message": "Processing anomaly detection in background"
}
```

### GET /anomalies

Returns recent anomaly alerts. Optional filter by engine_id.

```
GET /anomalies
GET /anomalies?engine_id=1
GET /anomalies?engine_id=1&limit=10
```

### GET /readings/{engine_id}

Returns all stored readings for a specific engine.

```
GET /readings/1
GET /readings/42
```

---

## Dataset

NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)

- 100 training engines run to failure
- 100 test engines cut off before failure
- 26 columns: engine ID, cycle, 3 operational settings, 21 sensors
- Single operating condition (sea level)
- Single fault mode (HPC degradation)
- 9 constant/near-zero variance sensors dropped
- 14 sensors used for anomaly detection

---

## Key Design Decisions

- One SSM per sensor — computationally efficient vs full multivariate SSM
- Weighted anomaly score — sensors weighted by RUL correlation strength
- 95th percentile threshold — balances detection rate vs false alarm rate
- BorderlineSMOTE not needed — unsupervised approach, no class labels
- MinMaxScaler chosen over StandardScaler — anomaly detection requires
  bounded 0-1 scale for comparable reconstruction errors
- SQLite for local development, PostgreSQL for Docker deployment —
  same code works in both environments via DATABASE_URL environment variable
- Background tasks — IoT sensors must not wait for processing

---

## Experiment Tracking

MLflow tracks two experiment runs:

Run 1 — SSM Training: logs model parameters, detection rates,
false alarm rates and threshold values across all 100 engines.

Run 2 — Test Evaluation: logs anomaly rates for high risk
(RUL < 50) vs low risk (RUL >= 50) test engines.

To view experiments locally:

```bash
cd notebooks
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open: http://127.0.0.1:5000

---

## Author

Tayyab
ML/AI Enginee
