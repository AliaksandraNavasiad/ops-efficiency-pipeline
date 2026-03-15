"""
test_transform.py
-----------------
Unit tests for the transform pipeline module.

Tests are designed to be fast, isolated, and self-contained — no file I/O,
no database, no network. Each test creates a minimal DataFrame fixture,
applies a single transform function, and asserts on the output.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.transform import (
    clean_trips,
    engineer_features,
    compute_driver_kpis,
    compute_zone_kpis,
    ZONE_EXPECTED_DURATION,
    DELAY_THRESHOLD_MULTIPLIER,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_minimal_trips(n: int = 10) -> pd.DataFrame:
    """Return a minimal valid trips DataFrame for testing."""
    rng = np.random.default_rng(0)
    zones = list(ZONE_EXPECTED_DURATION.keys())
    return pd.DataFrame({
        "trip_id": [f"T{i:04d}" for i in range(n)],
        "date": pd.date_range("2025-02-01", periods=n, freq="D"),
        "hour": rng.integers(0, 24, n),
        "city_zone": rng.choice(zones, n),
        "driver_id": [f"DRV_{i % 3:03d}" for i in range(n)],
        "distance_km": rng.uniform(1.5, 15.0, n).round(2),
        "duration_min": rng.uniform(10.0, 60.0, n).round(1),
        "fare_eur": rng.uniform(5.0, 40.0, n).round(2),
        "status": rng.choice(["completed", "cancelled"], n, p=[0.8, 0.2]),
        "cancellation_reason": [None] * n,
        "temperature_c": rng.uniform(0, 20, n),
        "precipitation_mm": rng.uniform(0, 5, n),
        "weather_condition": rng.choice(["clear", "light_rain", "heavy_rain"], n),
    })


# ── clean_trips tests ─────────────────────────────────────────────────────────

class TestCleanTrips:

    def test_date_cast_to_datetime(self):
        df = make_minimal_trips()
        df["date"] = df["date"].astype(str)  # simulate raw string input
        result = clean_trips(df)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_drops_null_trip_id(self):
        df = make_minimal_trips(5)
        df.loc[0, "trip_id"] = None
        result = clean_trips(df)
        assert len(result) == 4
        assert result["trip_id"].notna().all()

    def test_clips_negative_fare(self):
        df = make_minimal_trips(5)
        df.loc[0, "fare_eur"] = -10.0
        result = clean_trips(df)
        assert result["fare_eur"].min() >= 0.0

    def test_status_lowercased(self):
        df = make_minimal_trips(5)
        df["status"] = "COMPLETED"
        result = clean_trips(df)
        assert result["status"].str.islower().all()

    def test_output_row_count_unchanged_when_no_nulls(self):
        df = make_minimal_trips(10)
        result = clean_trips(df)
        assert len(result) == 10


# ── engineer_features tests ───────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_is_delayed_flag_correct(self):
        """A trip with duration well above threshold should be flagged delayed."""
        df = make_minimal_trips(2)
        df = clean_trips(df)
        # Force one trip to be massively over expected duration
        zone = df.iloc[0]["city_zone"]
        expected = ZONE_EXPECTED_DURATION[zone]
        df.at[df.index[0], "duration_min"] = expected * DELAY_THRESHOLD_MULTIPLIER + 10

        result = engineer_features(df)
        assert result.iloc[0]["is_delayed"] == True

    def test_no_delay_for_fast_trip(self):
        """A trip well under expected duration should not be flagged."""
        df = make_minimal_trips(2)
        df = clean_trips(df)
        zone = df.iloc[0]["city_zone"]
        expected = ZONE_EXPECTED_DURATION[zone]
        df.at[df.index[0], "duration_min"] = expected * 0.5

        result = engineer_features(df)
        assert result.iloc[0]["is_delayed"] == False

    def test_fare_per_min_positive(self):
        df = engineer_features(clean_trips(make_minimal_trips()))
        completed = df[df["status"] == "completed"]
        assert (completed["fare_per_min"] > 0).all()

    def test_peak_hour_flag(self):
        """Hour 8 (morning peak) should be flagged as peak."""
        df = make_minimal_trips(3)
        df = clean_trips(df)
        df["hour"] = 8
        result = engineer_features(df)
        assert result["is_peak_hour"].all()

    def test_off_peak_hour_flag(self):
        """Hour 3 (middle of night) should NOT be flagged as peak."""
        df = make_minimal_trips(3)
        df = clean_trips(df)
        df["hour"] = 3
        result = engineer_features(df)
        assert not result["is_peak_hour"].any()

    def test_new_columns_added(self):
        expected_cols = {
            "expected_duration_min", "is_delayed",
            "fare_per_min", "fare_per_km",
            "is_peak_hour", "day_of_week", "is_weekend",
        }
        result = engineer_features(clean_trips(make_minimal_trips()))
        assert expected_cols.issubset(set(result.columns))


# ── compute_driver_kpis tests ─────────────────────────────────────────────────

class TestComputeDriverKpis:

    def test_completion_rate_between_0_and_100(self):
        df = engineer_features(clean_trips(make_minimal_trips(30)))
        kpis = compute_driver_kpis(df)
        assert (kpis["completion_rate"] >= 0).all()
        assert (kpis["completion_rate"] <= 100).all()

    def test_delay_rate_between_0_and_1(self):
        df = engineer_features(clean_trips(make_minimal_trips(30)))
        kpis = compute_driver_kpis(df)
        assert (kpis["delay_rate"] >= 0).all()
        assert (kpis["delay_rate"] <= 1).all()

    def test_total_revenue_non_negative(self):
        df = engineer_features(clean_trips(make_minimal_trips(30)))
        kpis = compute_driver_kpis(df)
        assert (kpis["total_revenue_eur"] >= 0).all()

    def test_output_has_expected_columns(self):
        df = engineer_features(clean_trips(make_minimal_trips(30)))
        kpis = compute_driver_kpis(df)
        expected = {
            "driver_id", "completed_trips", "avg_fare_eur",
            "avg_fare_per_min", "delay_rate", "total_revenue_eur",
        }
        assert expected.issubset(set(kpis.columns))


# ── compute_zone_kpis tests ───────────────────────────────────────────────────

class TestComputeZoneKpis:

    def test_one_row_per_zone(self):
        df = engineer_features(clean_trips(make_minimal_trips(50)))
        kpis = compute_zone_kpis(df)
        # All zones present in input should appear in output
        input_zones = set(df[df["status"] == "completed"]["city_zone"].unique())
        output_zones = set(kpis["city_zone"].unique())
        assert input_zones == output_zones

    def test_peak_trip_share_between_0_and_1(self):
        df = engineer_features(clean_trips(make_minimal_trips(50)))
        kpis = compute_zone_kpis(df)
        assert (kpis["peak_trip_share"] >= 0).all()
        assert (kpis["peak_trip_share"] <= 1).all()
