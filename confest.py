import sys
import os
import pytest
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.structural import (
    UnobservedComponents
)

sys.path.insert(0, os.path.abspath('.'))

SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7',
    'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12',
    'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21'
]


def create_dummy_models():
    os.makedirs('models', exist_ok=True)
    np.random.seed(42)
    n = 100

    ssm_models = {}
    thresholds = {}

    for sensor in SENSORS:
        data = np.random.randn(n) * 0.1 + 0.4
        model = UnobservedComponents(
            data, level='local level'
        )
        result = model.fit(disp=False)
        errors = np.abs(
            data - result.fittedvalues.values
        )
        ssm_models[sensor] = result
        thresholds[sensor] = (
            errors.mean() + 3 * errors.std()
        )

    corr_weights = {s: 0.5 for s in SENSORS}
    overall_threshold = 0.05
    scaler = MinMaxScaler()
    dummy_data = np.random.randn(n, len(SENSORS))
    scaler.fit(dummy_data)

    joblib.dump(ssm_models, 'models/ssm_models.pkl')
    joblib.dump(thresholds, 'models/thresholds.pkl')
    joblib.dump(overall_threshold,
                'models/overall_threshold.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(corr_weights, 'models/corr_weights.pkl')
    joblib.dump(SENSORS, 'models/feature_columns.pkl')


create_dummy_models()


@pytest.fixture(autouse=True)
def reset_model_cache():
    import api.main as main
    main.ssm_models = None
    main.thresholds = None
    main.overall_threshold = None
    main.scaler = None
    main.corr_weights = None
    main.feature_columns = None
    yield
