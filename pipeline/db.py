"""
db.py
-----
SQLite database layer: schema creation, data loading, and analytical SQL queries.

Demonstrates:
- Schema design with appropriate types and constraints
- Bulk INSERT via pandas .to_sql()
- Analytical SQL: GROUP BY, JOIN, subqueries, CASE expressions,
  window functions (via SQLite 3.25+ support)
- Returning query results as DataFrames for downstream use

All queries are written as plain SQL strings to make skills clearly visible
to code reviewers — no ORM abstractions.
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/rides.db")


# ── Connection helper ──────────────────────────────────────────────────────────

def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


# ── Schema ─────────────────────────────────────────────────────────────────────

DDL_TRIPS = """
CREATE TABLE IF NOT EXISTS trips (
    trip_id              TEXT PRIMARY KEY,
    date                 TEXT NOT NULL,
    hour                 INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    city_zone            TEXT NOT NULL,
    driver_id            TEXT NOT NULL,
    distance_km          REAL NOT NULL CHECK (distance_km > 0),
    duration_min         REAL NOT NULL CHECK (duration_min > 0),
    fare_eur             REAL NOT NULL CHECK (fare_eur >= 0),
    status               TEXT NOT NULL CHECK (status IN ('completed', 'cancelled')),
    cancellation_reason  TEXT,
    temperature_c        REAL,
    precipitation_mm     REAL,
    weather_condition    TEXT,
    is_delayed           INTEGER,      -- 0 / 1 (SQLite has no BOOLEAN)
    fare_per_min         REAL,
    fare_per_km          REAL,
    is_peak_hour         INTEGER,
    day_of_week          TEXT,
    is_weekend           INTEGER
);
"""

DDL_WEATHER_HOURLY = """
CREATE TABLE IF NOT EXISTS weather_hourly (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    date             TEXT NOT NULL,
    hour             INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    temperature_c    REAL,
    precipitation_mm REAL,
    weather_condition TEXT,
    UNIQUE (date, hour)
);
"""

DDL_DRIVER_KPIS = """
CREATE TABLE IF NOT EXISTS driver_kpis (
    driver_id           TEXT PRIMARY KEY,
    completed_trips     INTEGER,
    total_trips         INTEGER,
    completion_rate     REAL,
    avg_fare_eur        REAL,
    avg_fare_per_min    REAL,
    delay_rate          REAL,
    total_revenue_eur   REAL
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't already exist."""
    cursor = conn.cursor()
    cursor.executescript(DDL_TRIPS + DDL_WEATHER_HOURLY + DDL_DRIVER_KPIS)
    conn.commit()
    print("[db] Schema created (or already exists).")


# ── Load ───────────────────────────────────────────────────────────────────────

def load_trips(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """
    Bulk-insert feature-engineered trips into the trips table.
    Replaces existing rows on conflict (idempotent re-runs).
    """
    cols = [
        "trip_id", "date", "hour", "city_zone", "driver_id",
        "distance_km", "duration_min", "fare_eur", "status", "cancellation_reason",
        "temperature_c", "precipitation_mm", "weather_condition",
        "is_delayed", "fare_per_min", "fare_per_km",
        "is_peak_hour", "day_of_week", "is_weekend",
    ]
    insert_df = df[[c for c in cols if c in df.columns]].copy()
    insert_df["date"] = insert_df["date"].astype(str)

    # Convert boolean columns to int for SQLite
    for bool_col in ["is_delayed", "is_peak_hour", "is_weekend"]:
        if bool_col in insert_df.columns:
            insert_df[bool_col] = insert_df[bool_col].astype(int)

    insert_df.to_sql("trips", conn, if_exists="replace", index=False)
    print(f"[db] Loaded {len(insert_df)} trips into trips table.")


def load_weather(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Bulk-insert hourly weather records."""
    cols = ["date", "hour", "temperature_c", "precipitation_mm", "weather_condition"]
    insert_df = df[cols].drop_duplicates(subset=["date", "hour"]).copy()
    insert_df.to_sql("weather_hourly", conn, if_exists="replace", index=False)
    print(f"[db] Loaded {len(insert_df)} weather records into weather_hourly.")


def load_driver_kpis(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """Bulk-insert driver KPI summary."""
    df.to_sql("driver_kpis", conn, if_exists="replace", index=False)
    print(f"[db] Loaded {len(df)} driver KPI records.")


# ── Analytical SQL Queries ─────────────────────────────────────────────────────

def query_completion_by_zone(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Completion rate, avg fare, and delay rate grouped by city zone.
    Uses CASE expressions to compute rate metrics inline.
    """
    sql = """
        SELECT
            city_zone,
            COUNT(*)                                                        AS total_trips,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END)          AS completed_trips,
            ROUND(
                100.0 * SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) / COUNT(*),
                1
            )                                                               AS completion_rate_pct,
            ROUND(AVG(CASE WHEN status = 'completed' THEN fare_eur END), 2) AS avg_fare_eur,
            ROUND(
                100.0 * AVG(CASE WHEN status = 'completed' THEN is_delayed END),
                1
            )                                                               AS delay_rate_pct
        FROM trips
        GROUP BY city_zone
        ORDER BY completion_rate_pct DESC;
    """
    return pd.read_sql_query(sql, conn)


def query_weather_impact(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Average fare-per-minute and delay rate by weather condition.
    Joins trips with weather_hourly to verify the enrichment join.
    """
    sql = """
        SELECT
            w.weather_condition,
            COUNT(t.trip_id)                                    AS trips,
            ROUND(AVG(t.fare_per_min), 3)                       AS avg_fare_per_min,
            ROUND(100.0 * AVG(t.is_delayed), 1)                 AS delay_rate_pct,
            ROUND(AVG(t.temperature_c), 1)                      AS avg_temp_c
        FROM trips t
        LEFT JOIN weather_hourly w
            ON t.date = w.date AND t.hour = w.hour
        WHERE t.status = 'completed'
          AND w.weather_condition IS NOT NULL
        GROUP BY w.weather_condition
        ORDER BY delay_rate_pct DESC;
    """
    return pd.read_sql_query(sql, conn)


def query_hourly_demand(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Trip volume and avg fare by hour of day — reveals demand peaks.
    """
    sql = """
        SELECT
            hour,
            COUNT(*)                            AS total_trips,
            SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_trips,
            ROUND(AVG(CASE WHEN status = 'completed' THEN fare_eur END), 2) AS avg_fare_eur,
            ROUND(100.0 * AVG(is_delayed), 1)   AS delay_rate_pct
        FROM trips
        GROUP BY hour
        ORDER BY hour;
    """
    return pd.read_sql_query(sql, conn)


def query_top_drivers(conn: sqlite3.Connection, top_n: int = 10) -> pd.DataFrame:
    """
    Top N drivers by revenue, with efficiency and reliability metrics.
    Uses a subquery to rank drivers.
    """
    sql = f"""
        SELECT
            driver_id,
            completed_trips,
            completion_rate,
            ROUND(avg_fare_eur, 2)      AS avg_fare_eur,
            ROUND(avg_fare_per_min, 3)  AS avg_fare_per_min,
            ROUND(delay_rate * 100, 1)  AS delay_rate_pct,
            ROUND(total_revenue_eur, 2) AS total_revenue_eur
        FROM driver_kpis
        ORDER BY total_revenue_eur DESC
        LIMIT {top_n};
    """
    return pd.read_sql_query(sql, conn)


def query_cancellation_reasons(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Breakdown of cancellation reasons — useful for root cause analysis.
    """
    sql = """
        SELECT
            COALESCE(cancellation_reason, 'not_cancelled') AS reason,
            COUNT(*)                                        AS count,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct_of_total
        FROM trips
        GROUP BY reason
        ORDER BY count DESC;
    """
    return pd.read_sql_query(sql, conn)


def print_summary(conn: sqlite3.Connection) -> None:
    """Print a quick summary of all analytical query results to stdout."""
    print("\n── Zone Completion & Delay ──────────────────────────────────")
    print(query_completion_by_zone(conn).to_string(index=False))

    print("\n── Weather Impact on Efficiency ─────────────────────────────")
    print(query_weather_impact(conn).to_string(index=False))

    print("\n── Top 10 Drivers by Revenue ────────────────────────────────")
    print(query_top_drivers(conn).to_string(index=False))

    print("\n── Cancellation Reason Breakdown ────────────────────────────")
    print(query_cancellation_reasons(conn).to_string(index=False))
