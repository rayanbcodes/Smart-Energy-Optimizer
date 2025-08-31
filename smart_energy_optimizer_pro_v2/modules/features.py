# modules/features.py
import pandas as pd

def create_time_features(df: pd.DataFrame, date_col='Date', hour_col='hour'):
    df = df.copy()
    df['datetime'] = pd.to_datetime(df[date_col]) + pd.to_timedelta(df[hour_col], unit='h')
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    return df

def rolling_features(df: pd.DataFrame, value_col='kwh', windows=[24,72]):
    df = df.copy().sort_values('datetime')
    for w in windows:
        df[f'roll_mean_{w}'] = df[value_col].rolling(window=w, min_periods=1).mean().fillna(method='bfill')
        df[f'roll_std_{w}'] = df[value_col].rolling(window=w, min_periods=1).std().fillna(0)
    return df

def make_supervised(df: pd.DataFrame, target='kwh', n_lags=24):
    df = df.copy().sort_values('datetime')
    for lag in range(1, n_lags+1):
        df[f'lag_{lag}'] = df[target].shift(lag).fillna(method='bfill')
    return df
