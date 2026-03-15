"""
transform.py
------------
All data cleaning, feature engineering, and KPI aggregation logic.

This module is intentionally stateless — every function takes a DataFrame
as input and returns a transformed DataFrame. This makes each step easy to
unit-test in isolation and reason about in code review.

Pipeline stages handled here:
1. clean_trips()        — type casting, null handling, validation
2. enrich_with_weather() — LEFT JOIN trips + weather on (date, hour)
3. engineer_features()  — derive delay flag, fare efficiency, peak flag
4. compute_driver_kpis() — GROUP BY driver → performance summary
5. compute_zone_kpis()  — GROUP BY city zone → demand + delay profile
"""

import pandas as pd
import numpy as np

# ── Expected duration benchmarks (minutes) per zone ──────────────────────────
# Used to flag trips that ran significantly over expected time.
ZONE_EXPECTED_DURATION: dict[str, float] = {
    "City Centre": 18.0,
    "Airport": 35.0,
    "Suburbs": 25.0,
    "University": 20.0,
    "Industrial": 30.0,
}

DELAY_THRESHOLD_MULTIPLIER = 1.3  # >30% over expected = delayed


# ── Stage 1: Clean ─────────────────────────────────────────────────────────────

def clean_trips(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean raw trip records.

    Operations:
    - Cast date to datetime, hour to int
    - Drop rows with null trip_id or driver_id
    - Clip fare and duration to plausible ranges
    - Standardise status to lowercase

    Parameters
    ----------
    df : pd.DataFrame
        Raw trips DataFrame from ingest.py.

    Returns
    -------
    pd.DataFrame
        Cleaned trips DataFrame.
    """
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df["hour"] = df["hour"].astype(int)
    df["status"] = df["status"].str.lower().str.strip()

    # Drop records missing critical identifiers
    before = len(df)
    df.dropna(subset=["trip_id", "driver_id"], inplace=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[transform] Dropped {dropped} rows with null IDs.")

    # Sanity-clip numeric fields
    df["fare_eur"] = df["fare_eur"].clip(lower=0.0)
    df["duration_min"] = df["duration_min"].clip(lower=1.0, upper=180.0)
    df["distance_km"] = df["distance_km"].clip(lower=0.1)

    print(f"[transform] clean_trips → {len(df)} records.")
    return df


# ── Stage 2: Enrich with Weather ───────────────────────────────────────────────

def enrich_with_weather(
    trips: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-join trip records with hourly weather on (date_str, hour).

    Trips without a matching weather record retain NaN weather columns,
    avoiding silent data loss.

    Parameters
    ----------
    trips : pd.DataFrame
        Cleaned trips DataFrame.
    weather : pd.DataFrame
        Weather DataFrame from api_client.py.

    Returns
    -------
    pd.DataFrame
        Trips enriched with temperature_c, precipitation_mm, weather_condition.
    """
    trips = trips.copy()
    trips["date_str"] = trips["date"].dt.strftime("%Y-%m-%d")

    weather_slim = weather[
        ["date", "hour", "temperature_c", "precipitation_mm", "weather_condition"]
    ].copy()

    enriched = trips.merge(
        weather_slim,
        left_on=["date_str", "hour"],
        right_on=["date", "hour"],
        how="left",
        suffixes=("", "_weather"),
    )

    # Drop the extra date column brought in by weather
    enriched.drop(columns=["date_weather"], errors="ignore", inplace=True)

    matched = enriched["temperature_c"].notna().sum()
    print(f"[transform] enrich_with_weather → {matched}/{len(enriched)} trips matched weather.")
    return enriched


# ── Stage 3: Feature Engineering ───────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive analytical features from cleaned and enriched trip data.

    New columns added:
    - expected_duration_min : zone benchmark (float)
    - is_delayed             : bool, duration > expected * DELAY_THRESHOLD
    - fare_per_min           : revenue efficiency metric (EUR/min)
    - fare_per_km            : distance efficiency metric (EUR/km)
    - is_peak_hour           : bool, trip in morning or evening peak
    - day_of_week            : weekday name
    - is_weekend             : bool

    Parameters
    ----------
    df : pd.DataFrame
        Enriched trips DataFrame.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame.
    """
    df = df.copy()

    # Expected duration per zone
    df["expected_duration_min"] = df["city_zone"].map(ZONE_EXPECTED_DURATION)

    # Delay flag: actual duration exceeds threshold multiple of expected
    df["is_delayed"] = (
        df["duration_min"] > df["expected_duration_min"] * DELAY_THRESHOLD_MULTIPLIER
    )

    # Efficiency metrics (guard against division by zero)
    df["fare_per_min"] = np.where(
        df["duration_min"] > 0,
        (df["fare_eur"] / df["duration_min"]).round(3),
        np.nan,
    )
    df["fare_per_km"] = np.where(
        df["distance_km"] > 0,
        (df["fare_eur"] / df["distance_km"]).round(3),
        np.nan,
    )

    # Peak hour: morning 07–09, evening 17–20
    df["is_peak_hour"] = df["hour"].between(7, 9) | df["hour"].between(17, 20)

    # Temporal features
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek >= 5

    print(f"[transform] engineer_features → {df.shape[1]} columns.")
    return df


# ── Stage 4: Driver KPIs ────────────────────────────────────────────────────────

def compute_driver_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trip-level data into per-driver performance KPIs.

    Metrics:
    - total_trips         : all trip records assigned
    - completed_trips     : status == 'completed'
    - completion_rate     : completed / total (%)
    - avg_fare_eur        : mean fare across completed trips
    - avg_fare_per_min    : mean revenue efficiency
    - delay_rate          : % of completed trips flagged as delayed
    - total_revenue_eur   : sum of fares (completed only)

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered trips DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per driver, sorted by total_revenue_eur descending.
    """
    completed = df[df["status"] == "completed"]

    driver_stats = (
        completed.groupby("driver_id")
        .agg(
            completed_trips=("trip_id", "count"),
            avg_fare_eur=("fare_eur", "mean"),
            avg_fare_per_min=("fare_per_min", "mean"),
            delay_rate=("is_delayed", "mean"),
            total_revenue_eur=("fare_eur", "sum"),
        )
        .reset_index()
    )

    # Merge total trips (including cancelled) for completion rate
    total = df.groupby("driver_id")["trip_id"].count().reset_index(name="total_trips")
    driver_stats = driver_stats.merge(total, on="driver_id", how="left")
    driver_stats["completion_rate"] = (
        driver_stats["completed_trips"] / driver_stats["total_trips"] * 100
    ).round(1)

    # Round float metrics
    for col in ["avg_fare_eur", "avg_fare_per_min", "delay_rate", "total_revenue_eur"]:
        driver_stats[col] = driver_stats[col].round(3)

    driver_stats.sort_values("total_revenue_eur", ascending=False, inplace=True)
    print(f"[transform] compute_driver_kpis → {len(driver_stats)} drivers profiled.")
    return driver_stats.reset_index(drop=True)


# ── Stage 5: Zone KPIs ─────────────────────────────────────────────────────────

def compute_zone_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate trip-level data into per-city-zone demand and delay profiles.

    Metrics:
    - total_trips         : all trip records
    - completion_rate     : % completed
    - avg_duration_min    : mean trip duration
    - avg_fare_eur        : mean fare
    - delay_rate          : % of completed trips delayed
    - peak_trip_share     : % of trips during peak hours

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered trips DataFrame.

    Returns
    -------
    pd.DataFrame
        One row per city zone.
    """
    completed = df[df["status"] == "completed"]

    zone_stats = (
        completed.groupby("city_zone")
        .agg(
            completed_trips=("trip_id", "count"),
            avg_duration_min=("duration_min", "mean"),
            avg_fare_eur=("fare_eur", "mean"),
            delay_rate=("is_delayed", "mean"),
            peak_trip_share=("is_peak_hour", "mean"),
        )
        .reset_index()
    )

    total = df.groupby("city_zone")["trip_id"].count().reset_index(name="total_trips")
    zone_stats = zone_stats.merge(total, on="city_zone", how="left")
    zone_stats["completion_rate"] = (
        zone_stats["completed_trips"] / zone_stats["total_trips"] * 100
    ).round(1)

    for col in ["avg_duration_min", "avg_fare_eur", "delay_rate", "peak_trip_share"]:
        zone_stats[col] = zone_stats[col].round(3)

    print(f"[transform] compute_zone_kpis → {len(zone_stats)} zones profiled.")
    return zone_stats
