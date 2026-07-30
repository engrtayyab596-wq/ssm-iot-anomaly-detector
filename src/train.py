import numpy as np
import joblib
import warnings
from statsmodels.tsa.statespace.structural import (
    UnobservedComponents
)
warnings.filterwarnings('ignore')


def train_ssm_models(healthy_data, final_sensors):
    ssm_models = {}
    thresholds = {}

    print("Training SSM for each sensor...")
    for sensor in final_sensors:
        sensor_data = healthy_data[sensor]
        model = UnobservedComponents(
            sensor_data,
            level='local level'
        )
        result = model.fit(disp=False)
        fitted = result.fittedvalues
        errors = np.abs(
            sensor_data.values - fitted.values
        )
        threshold = errors.mean() + 3 * errors.std()
        ssm_models[sensor] = result
        thresholds[sensor] = threshold
        print(f"  {sensor}: threshold={threshold:.4f}")

    return ssm_models, thresholds


def save_models(ssm_models, thresholds,
                overall_threshold, scaler,
                corr_weights, feature_columns,
                path='models'):
    joblib.dump(ssm_models,
                f'{path}/ssm_models.pkl')
    joblib.dump(thresholds,
                f'{path}/thresholds.pkl')
    joblib.dump(overall_threshold,
                f'{path}/overall_threshold.pkl')
    joblib.dump(scaler,
                f'{path}/scaler.pkl')
    joblib.dump(corr_weights,
                f'{path}/corr_weights.pkl')
    joblib.dump(feature_columns,
                f'{path}/feature_columns.pkl')
    print("All models saved successfully!")
