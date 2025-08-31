# modules/data_loader.py
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from modules.logger import get_logger

logger = get_logger(__name__)

def read_csv_auto(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(p)
    return df

def load_all(baseline_path: str = "data/baseline_fixed_load.csv",
             appliances_path: str = "data/appliances.csv",
             prices_path: str = "data/tou_prices.csv",
             history_path: Optional[str] = "data/hourly_sample_usage.csv"):
    baseline = read_csv_auto(baseline_path)
    appliances = read_csv_auto(appliances_path)
    prices = read_csv_auto(prices_path)
    history = read_csv_auto(history_path) if Path(history_path).exists() else None

    # cast types / validations
    baseline['hour'] = baseline['hour'].astype(int)
    baseline['kwh'] = baseline['kwh'].astype(float)

    prices['hour'] = prices['hour'].astype(int)
    prices['price_per_kwh'] = prices['price_per_kwh'].astype(float)

    # appliances: name,power_kw,duration_hours,flexible,earliest_start,latest_end
    for col in ['name','power_kw','duration_hours','flexible','earliest_start','latest_end']:
        if col not in appliances.columns:
            raise AssertionError(f"Appliances missing column: {col}")
    appliances['power_kw'] = appliances['power_kw'].astype(float)
    appliances['duration_hours'] = appliances['duration_hours'].astype(int)
    appliances['flexible'] = appliances['flexible'].astype(int)
    appliances['earliest_start'] = appliances['earliest_start'].astype(int)
    appliances['latest_end'] = appliances['latest_end'].astype(int)

    if history is not None:
        # expect Date,hour,kwh,temp optional
        history['hour'] = history['hour'].astype(int)
        history['kwh'] = history['kwh'].astype(float)
    logger.info("Loaded data successfully.")
    return baseline, appliances, prices, history
