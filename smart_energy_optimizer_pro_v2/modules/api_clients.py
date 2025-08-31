# modules/api_clients.py
import os
import requests
from typing import Optional, Dict, Any
from modules.logger import get_logger

logger = get_logger(__name__)
OPENWEATHER = os.getenv("OPENWEATHER_API_KEY")
EIA_KEY = os.getenv("EIA_API_KEY")

def fetch_weather_forecast(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Fetch hourly forecast (next 48 h) from OpenWeather One Call API if key present.
    Returns a dict or None if unavailable.
    """
    if not OPENWEATHER:
        logger.info("No OpenWeather API key configured; skipping weather fetch.")
        return None
    try:
        url = "https://api.openweathermap.org/data/2.5/onecall"
        params = {"lat": lat, "lon": lon, "exclude": "minutely,current,alerts", "units": "metric", "appid": OPENWEATHER}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.exception(f"Failed to fetch OpenWeather data: {e}")
        return None

def fetch_electricity_prices(region: str = "US") -> Optional[Dict[str, Any]]:
    """
    Try to use EIA API (if key provided) to fetch hourly electricity prices or series.
    If no API key or failure, return None.
    """
    if not EIA_KEY:
        logger.info("No EIA API key configured; skipping price fetch.")
        return None
    try:
        # Example: EIA series for real-time price varies by region; user must adjust series id.
        # We'll call a generic search endpoint as an example (this may need customizing per user).
        url = "https://api.eia.gov/series/"
        params = {"api_key": EIA_KEY, "series_id": "ELEC.PRICE."+region+"-HOURLY.M"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.exception(f"EIA fetch failed: {e}")
        return None
