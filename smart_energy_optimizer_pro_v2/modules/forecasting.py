# modules/forecasting.py
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from modules.logger import get_logger
from joblib import dump, load
import os

logger = get_logger(__name__)

MODEL_PATH = "models"
os.makedirs(MODEL_PATH, exist_ok=True)

def train_model(df: pd.DataFrame, feature_cols: List[str], target='kwh', test_frac=0.2, use_xgb=False) -> Dict:
    df = df.dropna().reset_index(drop=True)
    n_test = int(len(df) * test_frac)
    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]
    y_test = test[target]

    if use_xgb:
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=200, tree_method='hist', verbosity=0)
        except Exception as e:
            logger.info("XGBoost not available; falling back to RandomForest.")
            model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"Model trained. MAE={mae:.3f}, RMSE={rmse:.3f}")
    dump(model, os.path.join(MODEL_PATH, "rf_model.joblib"))
    return {"model": model, "mae": mae, "rmse": rmse, "X_test": X_test, "y_test": y_test, "preds": preds}

def load_model(path=None):
    p = path or os.path.join(MODEL_PATH, "rf_model.joblib")
    if not os.path.exists(p):
        raise FileNotFoundError("Model not trained yet.")
    return load(p)

def predict_horizon(model, recent_df: pd.DataFrame, features: List[str], horizon=24):
    """
    Predict next 'horizon' hours. recent_df should contain engineered features and lag_{k} columns.
    This function is iterative and updates lag features as it predicts.
    """
    out = []
    curr = recent_df.copy().reset_index(drop=True)
    for h in range(horizon):
        X = curr[features].iloc[[-1]]
        pred = model.predict(X)[0]
        next_dt = curr['datetime'].iloc[-1] + pd.Timedelta(hours=1)
        newrow = curr.iloc[[-1]].copy()
        newrow['datetime'] = [next_dt]
        newrow['kwh'] = pred
        # shift lag features
        for lag in range(1, 25):
            if f'lag_{lag}' in curr.columns:
                if lag == 1:
                    newrow['lag_1'] = curr['kwh'].iloc[-1]
                else:
                    newrow[f'lag_{lag}'] = curr[f'lag_{lag-1}'].iloc[-1]
        curr = pd.concat([curr, newrow], ignore_index=True)
        out.append({'datetime': next_dt, 'kwh': pred})
    return pd.DataFrame(out)
