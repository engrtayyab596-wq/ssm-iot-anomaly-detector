from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

sample_reading = {
    "engine_id": 1,
    "cycle": 1,
    "sensor_2": 0.44,
    "sensor_3": 0.39,
    "sensor_4": 0.41,
    "sensor_7": 0.58,
    "sensor_8": 0.27,
    "sensor_9": 0.17,
    "sensor_11": 0.36,
    "sensor_12": 0.62,
    "sensor_13": 0.29,
    "sensor_14": 0.21,
    "sensor_15": 0.41,
    "sensor_17": 0.40,
    "sensor_20": 0.56,
    "sensor_21": 0.58
}


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_sensor_reading_returns_received():
    response = client.post(
        '/sensor-reading',
        json=sample_reading
    )
    assert response.status_code == 200
    assert response.json()['status'] == 'received'
    assert response.json()['engine_id'] == 1


def test_sensor_reading_returns_instantly():
    response = client.post(
        '/sensor-reading',
        json=sample_reading
    )
    assert 'message' in response.json()
    assert 'background' in response.json()[
        'message'
    ].lower()


def test_anomalies_endpoint():
    response = client.get('/anomalies')
    assert response.status_code == 200
    assert 'total_alerts' in response.json()
    assert 'alerts' in response.json()


def test_readings_endpoint():
    response = client.get('/readings/1')
    assert response.status_code == 200
    assert response.json()['engine_id'] == 1
    assert 'readings' in response.json()
