import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath, columns):
    df = pd.read_csv(
        filepath,
        sep='\s+',
        header=None,
        names=columns,
        engine='python'
    )
    return df


def drop_constant_sensors(df):
    drop_cols = [
        'setting_1', 'setting_2', 'setting_3',
        'sensor_1', 'sensor_5', 'sensor_10',
        'sensor_16', 'sensor_18', 'sensor_19',
        'sensor_6'
    ]
    df = df.drop(drop_cols, axis=1)
    return df


def add_rul(df):
    max_cycles = df.groupby(
        'engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)
    return df


def normalise(df, sensors, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        df[sensors] = scaler.fit_transform(df[sensors])
    else:
        df[sensors] = scaler.transform(df[sensors])
    return df, scaler
