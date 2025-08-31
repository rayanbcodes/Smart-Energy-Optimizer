# tests/test_features.py
import pandas as pd
from modules.features import create_time_features, rolling_features, make_supervised

def test_feature_pipeline():
    data = pd.DataFrame({
        'Date': ['2025-01-01']*4 + ['2025-01-02']*4,
        'hour': [0,1,2,3,0,1,2,3],
        'kwh': [1.0, 1.2, 0.9, 1.1, 1.3, 1.0, 1.4, 1.2]
    })
    df = create_time_features(data)
    df = rolling_features(df, value_col='kwh', windows=[2])
    df = make_supervised(df, target='kwh', n_lags=2)
    assert 'lag_1' in df.columns and 'lag_2' in df.columns
