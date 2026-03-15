"""
api_client.py
-------------
Fetches real historical hourly weather data from the Open-Meteo API.
https://open-meteo.com/ — free, no API key required.

Retrieves temperature and precipitation for Kraków over the trip date range,
which is later used to enrich trip records and analyse weather impact on
ride efficiency and delay rates.

Demonstrates:
- HTTP GET with query parameters via `requests`
- JSON response parsing and normalisation
- Graceful error handling with retry logic
- Converting raw API payload to a clean Pandas DataFrame
"""

import time
import requests
import pandas as pd

# ── API Configuration ─────────────────────────────────────────────────────────

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Kraków, Poland coordinates
KRAKOW_LAT = 50.0647
KRAKOW_LON = 19.9450

DEFAULT_PARAMS = {
    "latitude": KRAKOW_LAT,
    "longitude": KRAKOW_LON,
    "hourly": "temperature_2m,precipitation",
    "timezone": "Europe/Warsaw",
}

MAX_RETRIES = 3
RETRY_DELAY_SEC = 2


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly weather data from Open-Meteo for the given date range.

    Parameters
    ----------
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        Hourly weather with columns: datetime, temperature_c, precipitation_mm, hour.

    Raises
    ------
    RuntimeError
        If the API call fails after MAX_RETRIES attempts.
    """
    params = {**DEFAULT_PARAMS, "start_date": start_date, "end_date": end_date}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[api_client] Fetching weather {start_date} → {end_date} "
                  f"(attempt {attempt}/{MAX_RETRIES})...")
            response = requests.get(BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            return _parse_response(response.json())

        except requests.exceptions.Timeout:
            print(f"[api_client] Request timed out. Retrying in {RETRY_DELAY_SEC}s...")
        except requests.exceptions.HTTPError as e:
            print(f"[api_client] HTTP error: {e}")
        except requests.exceptions.ConnectionError:
            print("[api_client] Connection error. Check network.")

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY_SEC)

    raise RuntimeError(
        f"[api_client] Failed to fetch weather after {MAX_RETRIES} attempts."
    )


def _parse_response(payload: dict) -> pd.DataFrame:
    """
    Parse raw Open-Meteo JSON payload into a clean DataFrame.

    Parameters
    ----------
    payload : dict
        Raw JSON response from the API.

    Returns
    -------
    pd.DataFrame
        Normalised hourly weather records.
    """
    hourly = payload.get("hourly", {})

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "precipitation_mm": hourly["precipitation"],
    })

    df["date"] = df["datetime"].dt.date.astype(str)
    df["hour"] = df["datetime"].dt.hour

    # Classify weather conditions for downstream grouping
    df["weather_condition"] = df["precipitation_mm"].apply(_classify_weather)

    print(f"[api_client] Retrieved {len(df)} hourly weather records.")
    return df


def _classify_weather(precip_mm: float) -> str:
    """
    Bucket precipitation into human-readable weather conditions.

    Parameters
    ----------
    precip_mm : float
        Millimetres of precipitation in a given hour.

    Returns
    -------
    str
        One of: 'clear', 'light_rain', 'heavy_rain'.
    """
    if precip_mm == 0:
        return "clear"
    elif precip_mm < 2.0:
        return "light_rain"
    else:
        return "heavy_rain"


def get_date_range(df: pd.DataFrame) -> tuple[str, str]:
    """
    Extract the min/max date range from a trips DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Trips DataFrame with a 'date' column.

    Returns
    -------
    tuple[str, str]
        (start_date, end_date) as 'YYYY-MM-DD' strings.
    """
    dates = pd.to_datetime(df["date"])
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")
