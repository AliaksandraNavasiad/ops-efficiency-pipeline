"""
dashboard.py
------------
Renders a 6-panel analytical KPI dashboard from the processed pipeline data.

Panels:
1. Trip Completion Rate by City Zone       (horizontal bar)
2. Delay Rate by Weather Condition         (bar with annotation)
3. Top 10 Drivers by Revenue               (horizontal bar, coloured by delay rate)
4. Hourly Demand Heatmap                   (line + fill, peak windows shaded)
5. Fare Distribution — Completed Trips     (histogram + KDE approximation)
6. Cancellation Reason Breakdown           (donut / pie)

Design principles:
- Dark background for readability in GitHub README previews
- Consistent colour palette (Uber-adjacent: black, white, green accent)
- Annotation on key insights — shows analytical storytelling
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sqlite3

OUTPUT_PATH = Path("output/dashboard.png")

# ── Colour palette ─────────────────────────────────────────────────────────────
BG = "#0a0a0a"
PANEL_BG = "#141414"
ACCENT = "#06C167"      # Uber green
ACCENT2 = "#FF6B35"     # orange for warnings/delays
TEXT = "#EEEEEE"
GRID = "#2a2a2a"
BAR_BASE = "#2E86AB"
BAR_ALT = "#A23B72"


def build_dashboard(conn: sqlite3.Connection, trips_df: pd.DataFrame) -> None:
    """
    Generate and save the 6-panel dashboard.

    Parameters
    ----------
    conn : sqlite3.Connection
        Open connection to the rides database (for SQL-driven panels).
    trips_df : pd.DataFrame
        Feature-engineered trips DataFrame (for distribution panels).
    """
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Import here to keep db dependency local
    from pipeline.db import (
        query_completion_by_zone,
        query_weather_impact,
        query_top_drivers,
        query_hourly_demand,
        query_cancellation_reasons,
    )

    zone_df = query_completion_by_zone(conn)
    weather_df = query_weather_impact(conn)
    drivers_df = query_top_drivers(conn, top_n=10)
    hourly_df = query_hourly_demand(conn)
    cancel_df = query_cancellation_reasons(conn)

    completed = trips_df[trips_df["status"] == "completed"]

    # ── Canvas ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.suptitle(
        "Ride-Hailing Operations — Efficiency & KPI Dashboard",
        fontsize=18, fontweight="bold", color=TEXT, y=0.98,
    )

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.06)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

    # ── Panel 1: Completion Rate by Zone ───────────────────────────────────────
    ax1 = axes[0]
    bars = ax1.barh(
        zone_df["city_zone"],
        zone_df["completion_rate_pct"],
        color=ACCENT, edgecolor="none", height=0.55,
    )
    ax1.set_xlim(0, 105)
    ax1.set_xlabel("Completion Rate (%)", color=TEXT, fontsize=9)
    ax1.set_title("Trip Completion Rate by Zone", color=TEXT, fontweight="bold", fontsize=11)
    ax1.xaxis.label.set_color(TEXT)
    ax1.tick_params(colors=TEXT)
    ax1.set_yticklabels(zone_df["city_zone"], color=TEXT)
    ax1.set_yticks(range(len(zone_df)))
    ax1.set_yticklabels(zone_df["city_zone"], color=TEXT)
    ax1.axvline(zone_df["completion_rate_pct"].mean(), color=ACCENT2,
                linestyle="--", linewidth=1.2, label="avg")
    ax1.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    for bar, val in zip(bars, zone_df["completion_rate_pct"]):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", color=TEXT, fontsize=8)
    ax1.grid(axis="x", color=GRID, linewidth=0.5)

    # ── Panel 2: Delay Rate by Weather Condition ───────────────────────────────
    ax2 = axes[1]
    weather_labels = weather_df["weather_condition"].str.replace("_", " ").str.title()
    colors_w = [ACCENT if w == "clear" else ACCENT2 if "heavy" in w else BAR_ALT
                for w in weather_df["weather_condition"]]
    bars2 = ax2.bar(
        weather_labels, weather_df["delay_rate_pct"],
        color=colors_w, edgecolor="none", width=0.5,
    )
    ax2.set_ylabel("Delay Rate (%)", color=TEXT, fontsize=9)
    ax2.set_title("Delay Rate by Weather Condition", color=TEXT, fontweight="bold", fontsize=11)
    ax2.tick_params(colors=TEXT)
    for bar, val in zip(bars2, weather_df["delay_rate_pct"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}%", ha="center", color=TEXT, fontsize=9, fontweight="bold")
    ax2.grid(axis="y", color=GRID, linewidth=0.5)

    # Insight annotation
    if len(weather_df) >= 2:
        max_row = weather_df.loc[weather_df["delay_rate_pct"].idxmax()]
        ax2.annotate(
            f"↑ {max_row['delay_rate_pct']:.0f}% delays\nin {max_row['weather_condition'].replace('_',' ')}",
            xy=(weather_labels[weather_df["delay_rate_pct"].idxmax()],
                max_row["delay_rate_pct"]),
            xytext=(0.65, 0.85), textcoords="axes fraction",
            color=ACCENT2, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1),
        )

    # ── Panel 3: Top 10 Drivers by Revenue ────────────────────────────────────
    ax3 = axes[2]
    d = drivers_df.head(10).sort_values("total_revenue_eur")
    norm_delay = (d["delay_rate_pct"] - d["delay_rate_pct"].min()) / (
        d["delay_rate_pct"].max() - d["delay_rate_pct"].min() + 1e-9
    )
    bar_colors = plt.cm.RdYlGn_r(norm_delay.values)
    ax3.barh(d["driver_id"], d["total_revenue_eur"],
             color=bar_colors, edgecolor="none", height=0.6)
    ax3.set_xlabel("Total Revenue (€)", color=TEXT, fontsize=9)
    ax3.set_title("Top 10 Drivers — Revenue vs Delay Rate\n(green=low delay, red=high delay)",
                  color=TEXT, fontweight="bold", fontsize=10)
    ax3.set_yticks(range(len(d)))
    ax3.set_yticklabels(d["driver_id"], color=TEXT, fontsize=8)
    ax3.grid(axis="x", color=GRID, linewidth=0.5)

    # ── Panel 4: Hourly Demand ────────────────────────────────────────────────
    ax4 = axes[3]
    ax4.fill_between(hourly_df["hour"], hourly_df["completed_trips"],
                     alpha=0.25, color=ACCENT)
    ax4.plot(hourly_df["hour"], hourly_df["completed_trips"],
             color=ACCENT, linewidth=2)
    # Shade peak windows
    ax4.axvspan(7, 9, alpha=0.12, color=ACCENT2, label="Morning peak")
    ax4.axvspan(17, 20, alpha=0.12, color=BAR_ALT, label="Evening peak")
    ax4.set_xlabel("Hour of Day", color=TEXT, fontsize=9)
    ax4.set_ylabel("Completed Trips", color=TEXT, fontsize=9)
    ax4.set_title("Hourly Demand Profile", color=TEXT, fontweight="bold", fontsize=11)
    ax4.set_xticks(range(0, 24, 2))
    ax4.tick_params(colors=TEXT)
    ax4.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    ax4.grid(color=GRID, linewidth=0.5)

    # ── Panel 5: Fare Distribution ────────────────────────────────────────────
    ax5 = axes[4]
    fares = completed["fare_eur"].dropna()
    ax5.hist(fares, bins=30, color=BAR_BASE, edgecolor=PANEL_BG, alpha=0.85)
    ax5.axvline(fares.mean(), color=ACCENT, linestyle="--", linewidth=1.5,
                label=f"Mean: €{fares.mean():.2f}")
    ax5.axvline(fares.median(), color=ACCENT2, linestyle="--", linewidth=1.5,
                label=f"Median: €{fares.median():.2f}")
    ax5.set_xlabel("Fare (€)", color=TEXT, fontsize=9)
    ax5.set_ylabel("Trip Count", color=TEXT, fontsize=9)
    ax5.set_title("Fare Distribution — Completed Trips", color=TEXT,
                  fontweight="bold", fontsize=11)
    ax5.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    ax5.grid(color=GRID, linewidth=0.5, axis="y")

    # ── Panel 6: Cancellation Reasons ────────────────────────────────────────
    ax6 = axes[5]
    cancel_only = cancel_df[cancel_df["reason"] != "not_cancelled"]
    if len(cancel_only) > 0:
        wedge_colors = [ACCENT2, BAR_ALT, BAR_BASE, "#888888"][:len(cancel_only)]
        wedges, texts, autotexts = ax6.pie(
            cancel_only["count"],
            labels=cancel_only["reason"].str.replace("_", "\n"),
            autopct="%1.0f%%",
            colors=wedge_colors,
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor=BG),  # donut
            textprops={"color": TEXT, "fontsize": 8},
        )
        for at in autotexts:
            at.set_color(BG)
            at.set_fontweight("bold")
    ax6.set_title("Cancellation Reason Breakdown", color=TEXT,
                  fontweight="bold", fontsize=11)

    # ── Footer ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.01,
        "Data: 500 synthetic trip records · Weather: Open-Meteo API (Kraków) · "
        "github.com/AliaksandraNavasiad/ops-efficiency-pipeline",
        ha="center", fontsize=7.5, color="#666666",
    )

    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"[dashboard] Dashboard saved → {OUTPUT_PATH}")
    plt.close()
