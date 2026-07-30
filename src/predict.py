import numpy as np
import joblib


def load_models(path='models'):
    ssm_models = joblib.load(
        f'{path}/ssm_models.pkl'
    )
    thresholds = joblib.load(
        f'{path}/thresholds.pkl'
    )
    overall_threshold = joblib.load(
        f'{path}/overall_threshold.pkl'
    )
    scaler = joblib.load(
        f'{path}/scaler.pkl'
    )
    corr_weights = joblib.load(
        f'{path}/corr_weights.pkl'
    )
    feature_columns = joblib.load(
        f'{path}/feature_columns.pkl'
    )
    return (ssm_models, thresholds,
            overall_threshold, scaler,
            corr_weights, feature_columns)


def compute_score(data, ssm_models,
                  corr_weights, feature_columns):
    sensor_scores = []
    for sensor in feature_columns:
        fitted_mean = (
            ssm_models[sensor].fittedvalues.mean()
        )
        actual_value = data[sensor]
        error = abs(actual_value - fitted_mean)
        weight = corr_weights[sensor]
        sensor_scores.append(error * weight)
    return float(np.mean(sensor_scores))
