"""
ingest.py
---------
Generates a reproducible synthetic dataset of ride-hailing trip records.

Each record simulates a single trip with:
- Temporal metadata (date, hour)
- Trip attributes (city zone, driver ID, distance, duration, fare)
- Operational flags (status, cancellation reason)

Using numpy.random with a fixed seed ensures results are identical across
machines — important for reproducibility in a shared codebase.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
N_TRIPS = 500
OUTPUT_PATH = Path("data/raw_trips.csv")

CITY_ZONES = ["City Centre", "Airport", "Suburbs", "University", "Industrial"]
STATUSES = ["completed", "completed", "completed", "cancelled", "cancelled"]  # weighted
CANCEL_REASONS = ["driver_no_show", "rider_cancelled", "no_driver_available", None]
DRIVERS = [f"DRV_{i:03d}" for i in range(1, 41)]  # 40 unique drivers

# Expected trip duration by zone (minutes) — used to derive delay flag later
ZONE_BASE_DURATION: dict[str, int] = {
    "City Centre": 18,
    "Airport": 35,
    "Suburbs": 25,
    "University": 20,
    "Industrial": 30,
}


def generate_trips(n: int = N_TRIPS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic ride-hailing trips DataFrame.

    Parameters
    ----------
    n : int
        Number of trip records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw trip records with columns:
        trip_id, date, hour, city_zone, driver_id, distance_km,
        duration_min, fare_eur, status, cancellation_reason.
    """
    rng = np.random.default_rng(seed)

    # Date range: last 30 days
    base_date = pd.Timestamp("2025-02-13")
    dates = [base_date + pd.Timedelta(days=int(d)) for d in rng.integers(0, 30, n)]

    # Hour distribution: bimodal (morning + evening peaks)
    peak_hours = np.concatenate([
        rng.integers(7, 10, int(n * 0.30)),   # morning peak
        rng.integers(17, 21, int(n * 0.35)),  # evening peak
        rng.integers(0, 24, n - int(n * 0.65)),  # off-peak
    ])
    rng.shuffle(peak_hours)

    zones = rng.choice(CITY_ZONES, n)
    drivers = rng.choice(DRIVERS, n)
    statuses = rng.choice(STATUSES, n)

    # Distance: varies by zone
    distance_km = np.round(
        rng.uniform(1.5, 8.0, n) + np.where(zones == "Airport", 10.0, 0.0), 2
    )

    # Duration: base by zone + noise + longer at peak hours
    base_durations = np.array([ZONE_BASE_DURATION[z] for z in zones])
    noise = rng.normal(0, 4, n)
    peak_penalty = np.where(
        (peak_hours >= 7) & (peak_hours <= 9) | (peak_hours >= 17) & (peak_hours <= 20),
        rng.uniform(2, 8, n),
        0,
    )
    duration_min = np.clip(
        np.round(base_durations + noise + peak_penalty, 1), 5, 90
    )

    # Fare: base rate + per-km + per-min surge during peaks
    surge = np.where(peak_penalty > 0, rng.uniform(1.1, 1.4, n), 1.0)
    fare_eur = np.round(
        (2.5 + distance_km * 0.9 + duration_min * 0.15) * surge, 2
    )

    # Cancellation reasons only for cancelled trips
    cancel_col = np.where(
        statuses == "cancelled",
        rng.choice(CANCEL_REASONS[:3], n),  # exclude None for cancelled
        None,
    )

    df = pd.DataFrame({
        "trip_id": [f"T{i:05d}" for i in range(1, n + 1)],
        "date": dates,
        "hour": peak_hours[:n],
        "city_zone": zones,
        "driver_id": drivers,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "fare_eur": fare_eur,
        "status": statuses,
        "cancellation_reason": cancel_col,
    })

    return df


def save_raw(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> None:
    """Persist raw trip data to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[ingest] Saved {len(df)} trip records → {path}")


if __name__ == "__main__":
    trips = generate_trips()
    save_raw(trips)
    print(trips.head())
