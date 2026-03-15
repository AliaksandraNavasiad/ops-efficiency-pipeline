"""
main.py
-------
Orchestrates the full ops-efficiency pipeline end-to-end.

Stages:
    1. Ingest      — generate synthetic trip records
    2. API Enrich  — fetch real weather from Open-Meteo
    3. Transform   — clean, feature-engineer, compute KPIs
    4. SQL Load    — persist to SQLite, run analytical queries
    5. Dashboard   — render 6-panel Matplotlib KPI figure

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.ingest import generate_trips, save_raw
from pipeline.api_client import fetch_weather, get_date_range
from pipeline.transform import (
    clean_trips,
    enrich_with_weather,
    engineer_features,
    compute_driver_kpis,
    compute_zone_kpis,
)
from pipeline.db import (
    get_connection,
    create_schema,
    load_trips,
    load_weather,
    load_driver_kpis,
    print_summary,
)
from analysis.dashboard import build_dashboard


def run_pipeline() -> None:
    print("\n" + "═" * 60)
    print("  OPS EFFICIENCY PIPELINE — starting")
    print("═" * 60)

    # ── 1. Ingest ──────────────────────────────────────────────────────────────
    print("\n[1/5] INGEST")
    raw_trips = generate_trips(n=500, seed=42)
    save_raw(raw_trips)

    # ── 2. API: Weather Enrichment ─────────────────────────────────────────────
    print("\n[2/5] API ENRICHMENT")
    start_date, end_date = get_date_range(raw_trips)
    weather_df = fetch_weather(start_date, end_date)

    # ── 3. Transform ───────────────────────────────────────────────────────────
    print("\n[3/5] TRANSFORM")
    cleaned = clean_trips(raw_trips)
    enriched = enrich_with_weather(cleaned, weather_df)
    featured = engineer_features(enriched)
    driver_kpis = compute_driver_kpis(featured)
    zone_kpis = compute_zone_kpis(featured)

    print(f"\n  Zone KPIs preview:\n{zone_kpis.to_string(index=False)}")

    # ── 4. SQL Load & Query ────────────────────────────────────────────────────
    print("\n[4/5] DATABASE")
    conn = get_connection()
    create_schema(conn)
    load_trips(conn, featured)
    load_weather(conn, weather_df)
    load_driver_kpis(conn, driver_kpis)
    print_summary(conn)

    # ── 5. Dashboard ───────────────────────────────────────────────────────────
    print("\n[5/5] DASHBOARD")
    build_dashboard(conn, featured)

    conn.close()

    print("\n" + "═" * 60)
    print("  PIPELINE COMPLETE")
    print("  → Database:  data/rides.db")
    print("  → Dashboard: output/dashboard.png")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()
