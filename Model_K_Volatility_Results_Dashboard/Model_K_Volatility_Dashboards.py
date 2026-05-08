from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Model_K.Model_K import (
    ASSUMPTIONS,
    HAS_PLOTLY,
    attach_outcomes,
    build_brier_decomposition,
    build_kalshi_reality_outcomes,
    build_metrics_summary,
    build_outcome_join_coverage,
    build_resolution_mismatches,
    build_sharpness,
    build_time_bucket_outputs,
    dataframe_to_html_table,
    default_settlement_csv,
    expanded_calibration_error,
    go,
    load_kalshi_price_outputs,
    load_settlements,
    make_subplots,
)
from Model_K_Volatility_Decomposition_RT.Model_K_Volatility_Decomposition_RT import (
    ROLLING_WINDOW_HOURS,
    ROLLING_WINDOW_MINUTES,
    SEGMENTS,
    add_realtime_window_thresholds,
    build_hourly_market_state_table,
    compute_hourly_realized_volatility,
    load_or_fetch_binance_minutes,
    thresholds_table,
    volatility_assumptions,
)


OUTPUT_FOLDER_NAME = "Model_K_Volatility_Dashboards_outputs"
TOTAL_LABEL = "Total Market"
VOLATILITY_LABEL = "Volatility Segment"

COLOR_TOTAL = "#6cb6ff"
COLOR_VOLATILITY = "#ff8f70"
COLOR_PERFECT = "#8ea3b7"
COLOR_GRID = "#22384a"
COLOR_AXIS = "#35516c"
COLOR_PAPER = "#0d1b26"
COLOR_BG = "#07131d"
COLOR_PANEL = "#10202d"
COLOR_PANEL_ALT = "#142736"
COLOR_TEXT = "#e8f1fa"
COLOR_MUTED = "#96aabd"

HIST_BINS = np.linspace(0.0, 1.0, 21)

REQUIRED_HOURLY_STATE_COLUMNS = [
    "event_ticker",
    "forecast_hour_start_utc",
    "realized_variance",
    "realized_volatility",
    "rolling_window_hours",
    "rolling_window_minutes",
    "rolling_window_observations",
    "rolling_percentile_rank",
    "primary_volatility_band",
    "is_low_volatility",
    "is_standard_volatility",
    "is_high_volatility",
    "is_low_volatility_extreme",
    "is_high_volatility_extreme",
    "q10_realized_volatility",
    "q25_realized_volatility",
    "q75_realized_volatility",
    "q90_realized_volatility",
]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / OUTPUT_FOLDER_NAME


def rt_output_root(root: Path) -> Path:
    return root / "Model_K_Volatility_Decomposition_RT" / "Model_K_Volatility_Decomposition_RT_outputs"


def legacy_vol_output_root(root: Path) -> Path:
    return root / "Model_K_Volatility_Decomposition" / "Model_K_Volatility_Decomposition_outputs"


def first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path and path.exists():
            return path
    return None


def default_hourly_states_csv(root: Path) -> Optional[Path]:
    return first_existing(
        [
            rt_output_root(root) / "hourly_market_volatility_segments.csv",
            legacy_vol_output_root(root) / "hourly_market_volatility_segments.csv",
        ]
    )


def default_binance_cache_csv(root: Path) -> Optional[Path]:
    return first_existing(
        [
            rt_output_root(root) / "binance_1m_klines.csv",
            legacy_vol_output_root(root) / "binance_1m_klines.csv",
        ]
    )


def resolve_path(root: Path, value: Path) -> Path:
    return value if value.is_absolute() else (root / value)


def resolve_optional_path(root: Path, value: Optional[Path]) -> Optional[Path]:
    if value is None:
        return None
    return resolve_path(root, value)


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Build dark-mode comparison dashboards for baseline Model K versus each real-time "
            "volatility segment from Model_K_Volatility_Decomposition_RT."
        )
    )
    parser.add_argument(
        "--kalshi-price-dir",
        type=Path,
        default=root / "Data_Sourcing" / "Kalshi_Pricing_Fetch" / "hourly_events_price_data",
        help="Folder containing the minute-level Kalshi hourly pricing CSVs.",
    )
    parser.add_argument(
        "--settlement-csv",
        type=Path,
        default=default_settlement_csv(root),
        help="Settlement CSV with official Kalshi outcomes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir(),
        help="Root directory for the comparison dashboard outputs.",
    )
    parser.add_argument(
        "--hourly-market-states-csv",
        type=Path,
        default=default_hourly_states_csv(root),
        help=(
            "Optional precomputed hourly_market_volatility_segments.csv. "
            "If present and it covers the scored events, it is reused."
        ),
    )
    parser.add_argument(
        "--binance-minute-cache-csv",
        type=Path,
        default=default_binance_cache_csv(root),
        help=(
            "Optional Binance 1-minute cache used when the hourly market state table "
            "must be recomputed."
        ),
    )
    parser.add_argument(
        "--refresh-binance-cache",
        action="store_true",
        help="Force a fresh Binance minute download if the hourly state table must be recomputed.",
    )
    parser.add_argument(
        "--skip-binance-audit",
        action="store_true",
        help="Skip the diagnostic Binance-vs-Kalshi resolution audit tables.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of calibration bins.")
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Threshold used for binary classification accuracy tables.",
    )
    return parser.parse_args()


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"}).fillna(False)


def load_hourly_market_states_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in ["forecast_hour_start_utc", "hour_start_utc"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], utc=True)
    for column in REQUIRED_HOURLY_STATE_COLUMNS:
        if column.startswith("is_") and column in df.columns:
            df[column] = coerce_bool(df[column])
    return df


def validate_hourly_market_states(hourly_market_states: pd.DataFrame, raw: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_HOURLY_STATE_COLUMNS if column not in hourly_market_states.columns]
    if missing:
        raise ValueError(
            "Hourly market state table is missing required columns: "
            f"{', '.join(sorted(missing))}."
        )

    required_events = set(raw["event_ticker"].astype(str))
    available_events = set(hourly_market_states["event_ticker"].astype(str))
    missing_events = sorted(required_events - available_events)
    if missing_events:
        sample = ", ".join(missing_events[:5])
        raise ValueError(
            "Hourly market state table does not cover all scored events. "
            f"Example missing event_ticker(s): {sample}"
        )


def compute_or_load_hourly_market_states(
    *,
    raw: pd.DataFrame,
    output_root: Path,
    root: Path,
    hourly_market_states_csv: Optional[Path],
    binance_minute_cache_csv: Optional[Path],
    refresh_binance_cache: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    notes: List[str] = []

    preferred_hourly_states = resolve_optional_path(root, hourly_market_states_csv)
    if preferred_hourly_states and preferred_hourly_states.exists():
        hourly_market_states = load_hourly_market_states_csv(preferred_hourly_states)
        validate_hourly_market_states(hourly_market_states, raw)
        notes.append(f"Reused hourly volatility segment table from {preferred_hourly_states}.")
        thresholds = thresholds_table(hourly_market_states)
        return hourly_market_states, thresholds, notes

    required_start = raw["event_datetime_utc"].min() - pd.Timedelta(hours=1 + ROLLING_WINDOW_HOURS)
    required_end = raw["event_datetime_utc"].max()

    preferred_cache = resolve_optional_path(root, binance_minute_cache_csv)
    cache_path = preferred_cache or (output_root / "binance_1m_klines.csv")
    cache_path = cache_path if cache_path.is_absolute() else (root / cache_path)

    binance_minutes = load_or_fetch_binance_minutes(
        cache_path=cache_path,
        start_utc=required_start,
        end_utc=required_end,
        refresh_cache=refresh_binance_cache,
    )
    hourly_volatility = compute_hourly_realized_volatility(binance_minutes)
    hourly_volatility = add_realtime_window_thresholds(hourly_volatility)
    hourly_market_states = build_hourly_market_state_table(raw, hourly_volatility)
    thresholds = thresholds_table(hourly_market_states)

    binance_minutes.to_csv(output_root / "binance_1m_klines.csv", index=False)
    hourly_volatility.to_csv(output_root / "binance_hourly_realized_volatility.csv", index=False)
    notes.append(f"Computed hourly volatility segment table using Binance minute cache {cache_path}.")
    return hourly_market_states, thresholds, notes


def slice_by_event_ids(frame: pd.DataFrame, event_ids: set[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[frame["event_ticker"].astype(str).isin(event_ids)].copy()


def build_bundle(
    *,
    name: str,
    raw: pd.DataFrame,
    forecasts: pd.DataFrame,
    unmatched: pd.DataFrame,
    calibration_bins: int,
    classification_threshold: float,
    skip_binance_audit: bool,
    n_hourly_markets: int,
) -> Dict[str, Any]:
    coverage = build_outcome_join_coverage(forecasts, raw, unmatched)
    resolution_mismatches = pd.DataFrame() if skip_binance_audit else build_resolution_mismatches(raw)
    metrics = build_metrics_summary(raw, threshold=classification_threshold)
    decomposition = build_brier_decomposition(raw, bins=calibration_bins)
    calibration, expanded_summary = expanded_calibration_error(raw, bins=calibration_bins)
    sharpness = build_sharpness(raw)
    time_bucket_metrics, time_bucket_accuracy, time_bucket_brier, time_bucket_calibration = build_time_bucket_outputs(
        raw,
        calibration_bins=calibration_bins,
        threshold=classification_threshold,
    )

    overall_metrics = metrics.loc[metrics["segment"] == "overall"].iloc[0]
    overall_ece = expanded_summary.loc[expanded_summary["segment"] == "overall"].iloc[0]

    return {
        "name": name,
        "raw": raw,
        "forecasts": forecasts,
        "unmatched": unmatched,
        "coverage": coverage,
        "resolution_mismatches": resolution_mismatches,
        "metrics": metrics,
        "decomposition": decomposition,
        "calibration": calibration,
        "expanded_summary": expanded_summary,
        "sharpness": sharpness,
        "time_bucket_metrics": time_bucket_metrics,
        "time_bucket_accuracy": time_bucket_accuracy,
        "time_bucket_brier": time_bucket_brier,
        "time_bucket_calibration": time_bucket_calibration,
        "n_hourly_markets": int(n_hourly_markets),
        "overall_metrics": overall_metrics,
        "overall_ece": overall_ece,
    }


def overview_comparison_table(total_bundle: Dict[str, Any], segment_bundle: Dict[str, Any]) -> pd.DataFrame:
    total_overall = total_bundle["overall_metrics"]
    segment_overall = segment_bundle["overall_metrics"]
    total_ece = total_bundle["overall_ece"]
    segment_ece = segment_bundle["overall_ece"]

    rows = [
        ("Scored rows", float(segment_overall["n_forecasts"]), float(total_overall["n_forecasts"])),
        ("Event contracts", float(segment_overall["n_event_contracts"]), float(total_overall["n_event_contracts"])),
        ("Brier score", float(segment_overall["brier_score"]), float(total_overall["brier_score"])),
        ("Log loss", float(segment_overall["log_loss"]), float(total_overall["log_loss"])),
        ("ECE", float(segment_ece["expected_calibration_error"]), float(total_ece["expected_calibration_error"])),
    ]
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "volatility": segment_value,
                "total": total_value,
                "delta": segment_value - total_value,
            }
            for metric, segment_value, total_value in rows
        ]
    )


def build_comparison_table(
    *,
    total_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    sort_cols: Optional[Sequence[str]] = None,
    hidden_key_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    hidden = set(hidden_key_cols or [])
    total = total_df[list(dict.fromkeys([*key_cols, *value_cols]))].copy()
    segment = segment_df[list(dict.fromkeys([*key_cols, *value_cols]))].copy()
    merged = segment.merge(total, on=list(key_cols), how="outer", suffixes=("_volatility", "_total"))

    ordered_cols: List[str] = [column for column in key_cols if column not in hidden]
    for column in value_cols:
        vol_col = f"{column}_volatility"
        total_col = f"{column}_total"
        ordered_cols.extend([vol_col, total_col])
        if pd.api.types.is_numeric_dtype(segment[column]) and pd.api.types.is_numeric_dtype(total[column]):
            delta_col = f"{column}_delta"
            merged[delta_col] = merged[vol_col] - merged[total_col]
            ordered_cols.append(delta_col)

    if sort_cols:
        merged = merged.sort_values(list(sort_cols), kind="stable")

    ordered_cols = [column for column in ordered_cols if column in merged.columns]
    return merged[ordered_cols].reset_index(drop=True)


def threshold_context_table(thresholds: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "threshold_name",
        "percentile",
        "window_hours",
        "latest_realized_volatility_cutoff",
        "median_realized_volatility_cutoff",
        "min_realized_volatility_cutoff",
        "max_realized_volatility_cutoff",
    ]
    return thresholds[columns].copy()


def plotly_theme(
    fig: Any,
    *,
    height: int,
    barmode: Optional[str] = None,
    legend_y: float = 1.12,
    top_margin: int = 76,
) -> None:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLOR_PAPER,
        plot_bgcolor=COLOR_PAPER,
        font=dict(color=COLOR_TEXT),
        legend=dict(
            orientation="h",
            x=0.0,
            y=legend_y,
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        margin=dict(l=56, r=28, t=top_margin, b=48),
        height=height,
    )
    if barmode:
        fig.update_layout(barmode=barmode)
    fig.update_xaxes(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zerolinecolor=COLOR_AXIS)
    fig.update_yaxes(gridcolor=COLOR_GRID, linecolor=COLOR_AXIS, zerolinecolor=COLOR_AXIS)


def comparison_charts_html(total_bundle: Dict[str, Any], segment_bundle: Dict[str, Any], segment_name: str) -> tuple[str, str]:
    if not HAS_PLOTLY:
        fallback = (
            "<div class='note-card'>"
            "<p>Interactive Plotly charts were unavailable in this environment, so only the comparison tables were generated.</p>"
            "</div>"
        )
        return fallback, fallback

    total_calibration = total_bundle["calibration"]
    total_calibration = total_calibration[
        (total_calibration["segment"] == "overall") & (total_calibration["n_forecasts"] > 0)
    ].copy()
    segment_calibration = segment_bundle["calibration"]
    segment_calibration = segment_calibration[
        (segment_calibration["segment"] == "overall") & (segment_calibration["n_forecasts"] > 0)
    ].copy()

    overlay = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calibration Curve Overlay", "Probability Distribution Overlay"),
        horizontal_spacing=0.11,
    )
    overlay.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=COLOR_PERFECT, dash="dash"),
            name="Perfect calibration",
        ),
        row=1,
        col=1,
    )
    overlay.add_trace(
        go.Scatter(
            x=total_calibration["mean_p_kalshi"],
            y=total_calibration["observed_frequency"],
            mode="lines+markers",
            name=TOTAL_LABEL,
            marker=dict(color=COLOR_TOTAL, size=8),
            line=dict(color=COLOR_TOTAL, width=3),
        ),
        row=1,
        col=1,
    )
    overlay.add_trace(
        go.Scatter(
            x=segment_calibration["mean_p_kalshi"],
            y=segment_calibration["observed_frequency"],
            mode="lines+markers",
            name=segment_name,
            marker=dict(color=COLOR_VOLATILITY, size=8),
            line=dict(color=COLOR_VOLATILITY, width=3),
        ),
        row=1,
        col=1,
    )
    overlay.add_trace(
        go.Histogram(
            x=total_bundle["raw"]["p_kalshi"],
            xbins=dict(start=0.0, end=1.0, size=0.05),
            opacity=0.60,
            marker_color=COLOR_TOTAL,
            name=TOTAL_LABEL,
        ),
        row=1,
        col=2,
    )
    overlay.add_trace(
        go.Histogram(
            x=segment_bundle["raw"]["p_kalshi"],
            xbins=dict(start=0.0, end=1.0, size=0.05),
            opacity=0.60,
            marker_color=COLOR_VOLATILITY,
            name=segment_name,
        ),
        row=1,
        col=2,
    )
    overlay.update_xaxes(range=[0, 1], title_text="Mean Kalshi probability", row=1, col=1)
    overlay.update_yaxes(range=[0, 1], title_text="Observed frequency", row=1, col=1)
    overlay.update_xaxes(range=[0, 1], title_text="Kalshi probability", row=1, col=2)
    overlay.update_yaxes(title_text="Count", row=1, col=2)
    overlay.update_annotations(y=1.03)
    plotly_theme(overlay, height=430, barmode="overlay", legend_y=1.18, top_margin=108)

    total_brier = total_bundle["decomposition"].loc[total_bundle["decomposition"]["segment"] == "overall"].iloc[0]
    segment_brier = segment_bundle["decomposition"].loc[segment_bundle["decomposition"]["segment"] == "overall"].iloc[0]
    categories = [
        ("Brier Score", "brier_score"),
        ("Reliability", "reliability"),
        ("Resolution", "resolution"),
        ("Uncertainty", "uncertainty"),
        ("Brier From Decomp", "brier_from_decomposition"),
    ]

    brier_chart = go.Figure()
    brier_chart.add_trace(
        go.Bar(
            x=[label for label, _ in categories],
            y=[float(total_brier[column]) for _, column in categories],
            name=TOTAL_LABEL,
            marker_color=COLOR_TOTAL,
        )
    )
    brier_chart.add_trace(
        go.Bar(
            x=[label for label, _ in categories],
            y=[float(segment_brier[column]) for _, column in categories],
            name=segment_name,
            marker_color=COLOR_VOLATILITY,
        )
    )
    brier_chart.update_yaxes(title_text="Metric value")
    plotly_theme(brier_chart, height=360, barmode="group")

    return (
        overlay.to_html(full_html=False, include_plotlyjs="cdn"),
        brier_chart.to_html(full_html=False, include_plotlyjs=False),
    )


def format_number(value: Any, *, digits: int = 4, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"

    value_float = float(value)
    prefix = "+" if signed and value_float >= 0 else ""
    return f"{prefix}{value_float:,.{digits}f}"


def format_value_for_column(column: str, value: Any) -> str:
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d %H:%M UTC")
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if not isinstance(value, (np.integer, int, np.floating, float)) or pd.isna(value):
        return "n/a" if pd.isna(value) else str(value)

    lower = column.lower()
    if lower.startswith("delta") or lower.endswith("_delta"):
        digits = 2 if "pct" in lower else 4
        return format_number(value, digits=digits, signed=True)
    if lower.startswith("n_") or any(
        token in lower
        for token in ["rows", "contracts", "correct", "incorrect", "window_hours", "window_minutes", "minute_bars", "mismatch"]
    ):
        return format_number(int(round(float(value))), digits=0)
    if "pct" in lower:
        return format_number(value, digits=2)
    return format_number(value, digits=4)


def format_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().astype(object)
    if "metric" in out.columns and {"volatility", "total", "delta"}.issubset(out.columns):
        count_metrics = {"scored rows", "event contracts"}
        for index, metric in out["metric"].astype(str).str.lower().items():
            if metric in count_metrics:
                out.at[index, "volatility"] = format_number(out.at[index, "volatility"], digits=0)
                out.at[index, "total"] = format_number(out.at[index, "total"], digits=0)
                out.at[index, "delta"] = format_number(out.at[index, "delta"], digits=0, signed=True)
            else:
                out.at[index, "volatility"] = format_number(out.at[index, "volatility"], digits=4)
                out.at[index, "total"] = format_number(out.at[index, "total"], digits=4)
                out.at[index, "delta"] = format_number(out.at[index, "delta"], digits=4, signed=True)
    for column in out.columns:
        out[column] = out[column].map(lambda value, col=column: format_value_for_column(col, value))
    return out


def render_table(df: pd.DataFrame, *, max_rows: Optional[int] = None) -> str:
    return dataframe_to_html_table(format_table(df), max_rows=max_rows)


def side_by_side_table_html(left_title: str, left_df: pd.DataFrame, right_title: str, right_df: pd.DataFrame) -> str:
    return (
        "<div class='table-grid'>"
        f"<section><h3>{html.escape(left_title)}</h3><div class='table-wrap'>{render_table(left_df, max_rows=25)}</div></section>"
        f"<section><h3>{html.escape(right_title)}</h3><div class='table-wrap'>{render_table(right_df, max_rows=25)}</div></section>"
        "</div>"
    )


def dashboard_style() -> str:
    return f"""<style>
    :root {{
      color-scheme: dark;
      --bg: {COLOR_BG};
      --bg-2: #0b1823;
      --panel: {COLOR_PANEL};
      --panel-2: {COLOR_PANEL_ALT};
      --line: #22394b;
      --line-soft: rgba(255,255,255,0.05);
      --ink: {COLOR_TEXT};
      --muted: {COLOR_MUTED};
      --total: {COLOR_TOTAL};
      --vol: {COLOR_VOLATILITY};
      --pill: #122736;
    }}
    html {{
      background: var(--bg);
    }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(108, 182, 255, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(255, 143, 112, 0.15), transparent 28%),
        linear-gradient(180deg, var(--bg-2) 0%, var(--bg) 100%);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1260px;
      margin: 0 auto;
      padding: 34px 24px 54px;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }}
    h2 {{
      margin: 36px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 15px;
      color: var(--ink);
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.5;
    }}
    a {{
      color: #9aceff;
    }}
    .hero {{
      padding: 24px 24px 20px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)),
        var(--panel);
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.05),
        0 16px 40px rgba(0,0,0,0.18);
    }}
    .eyebrow {{
      margin: 0 0 8px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #a9c7e2;
    }}
    .lead {{
      max-width: 920px;
      margin: 14px 0 0;
      font-size: 15px;
    }}
    .hero-pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--pill);
      font-size: 13px;
      color: var(--ink);
    }}
    .swatch {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .swatch.total {{
      background: var(--total);
    }}
    .swatch.vol {{
      background: var(--vol);
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01)),
        var(--panel);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
      padding: 20px 20px 18px;
    }}
    .note-card {{
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel-2);
    }}
    .note-card p {{
      margin: 0;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.01);
    }}
    .table-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
      background: transparent;
    }}
    .data-table th,
    .data-table td {{
      padding: 10px 11px;
      text-align: left;
      border-bottom: 1px solid var(--line);
    }}
    .data-table th {{
      background: var(--panel-2);
      color: var(--ink);
      font-weight: 650;
      position: sticky;
      top: 0;
    }}
    .data-table td {{
      color: var(--ink);
    }}
    .section-stack {{
      display: grid;
      gap: 18px;
    }}
    .chart-wrap {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel);
      overflow: hidden;
    }}
    .chart-note {{
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }}
    .links {{
      display: grid;
      gap: 8px;
      padding-left: 18px;
    }}
    code {{
      background: rgba(255,255,255,0.04);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 2px 5px;
      color: var(--ink);
    }}
  </style>"""


def build_segment_dashboard_html(
    *,
    segment_name: str,
    segment_rule: str,
    total_bundle: Dict[str, Any],
    segment_bundle: Dict[str, Any],
    total_hourly_markets: int,
    thresholds: pd.DataFrame,
    assumptions: Sequence[str],
    notes: Sequence[str],
) -> str:
    overview = overview_comparison_table(total_bundle, segment_bundle)
    overlay_chart_html, brier_chart_html = comparison_charts_html(total_bundle, segment_bundle, segment_name)

    metrics_comparison = build_comparison_table(
        total_df=total_bundle["metrics"],
        segment_df=segment_bundle["metrics"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "n_event_contracts",
            "base_rate",
            "mean_p_kalshi",
            "brier_score",
            "log_loss",
            "classification_accuracy",
        ],
    )
    decomposition_comparison = build_comparison_table(
        total_df=total_bundle["decomposition"],
        segment_df=segment_bundle["decomposition"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "brier_score",
            "reliability",
            "resolution",
            "uncertainty",
            "brier_from_decomposition",
        ],
    )
    calibration_comparison = build_comparison_table(
        total_df=total_bundle["expanded_summary"],
        segment_df=segment_bundle["expanded_summary"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "expected_calibration_error",
            "root_mean_squared_calibration_error",
            "max_calibration_error",
        ],
    )
    sharpness_comparison = build_comparison_table(
        total_df=total_bundle["sharpness"],
        segment_df=segment_bundle["sharpness"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "base_rate",
            "mean_p_kalshi",
            "forecast_std",
            "sharpness_variance_from_base_rate",
            "mean_abs_distance_from_0_5",
            "mean_predictive_variance_p_times_1_minus_p",
        ],
    )
    coverage_comparison = build_comparison_table(
        total_df=total_bundle["coverage"],
        segment_df=segment_bundle["coverage"],
        key_cols=["scope", "value"],
        value_cols=["total_forecast_rows", "matched_rows", "unmatched_rows", "match_rate"],
    )

    isolated_metrics_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_metrics"][
            (total_bundle["time_bucket_metrics"]["bucket_type"] == "isolated")
            & (total_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        segment_df=segment_bundle["time_bucket_metrics"][
            (segment_bundle["time_bucket_metrics"]["bucket_type"] == "isolated")
            & (segment_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "n_event_contracts",
            "base_rate",
            "mean_p_kalshi",
            "brier_score",
            "log_loss",
            "classification_accuracy",
        ],
    )
    decile_metrics_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_metrics"][
            (total_bundle["time_bucket_metrics"]["bucket_type"] == "decile")
            & (total_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        segment_df=segment_bundle["time_bucket_metrics"][
            (segment_bundle["time_bucket_metrics"]["bucket_type"] == "decile")
            & (segment_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "n_event_contracts",
            "base_rate",
            "mean_p_kalshi",
            "brier_score",
            "log_loss",
            "classification_accuracy",
        ],
    )
    time_bucket_accuracy_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_accuracy"][total_bundle["time_bucket_accuracy"]["n_forecasts"] > 0],
        segment_df=segment_bundle["time_bucket_accuracy"][segment_bundle["time_bucket_accuracy"]["n_forecasts"] > 0],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "correct_forecasts",
            "incorrect_forecasts",
            "classification_accuracy",
            "classification_accuracy_pct",
            "threshold",
        ],
    )
    time_bucket_brier_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_brier"][total_bundle["time_bucket_brier"]["n_forecasts"] > 0],
        segment_df=segment_bundle["time_bucket_brier"][segment_bundle["time_bucket_brier"]["n_forecasts"] > 0],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "brier_score",
            "reliability",
            "resolution",
            "uncertainty",
            "brier_from_decomposition",
        ],
    )

    total_mismatches = total_bundle["resolution_mismatches"]
    segment_mismatches = segment_bundle["resolution_mismatches"]
    mismatch_count = pd.DataFrame(
        [
            {"scope": segment_name, "audit_mismatches": len(segment_mismatches)},
            {"scope": TOTAL_LABEL, "audit_mismatches": len(total_mismatches)},
        ]
    )

    assumption_html = "".join(f"<li>{html.escape(item)}</li>" for item in assumptions)
    notes_html = "".join(f"<li>{html.escape(item)}</li>" for item in notes)

    segment_share = (
        float(segment_bundle["n_hourly_markets"]) / float(total_hourly_markets)
        if total_hourly_markets
        else np.nan
    )
    segment_match_rate = segment_bundle["coverage"].loc[
        segment_bundle["coverage"]["scope"] == "overall", "match_rate"
    ].iloc[0]
    total_match_rate = total_bundle["coverage"].loc[
        total_bundle["coverage"]["scope"] == "overall", "match_rate"
    ].iloc[0]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model K Comparison: {html.escape(segment_name)} vs Total Market</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Model K Comparison Dashboard</p>
    <h1>{html.escape(segment_name)} vs Total Market</h1>
    <p class="lead">
      This page compares the baseline <code>Model_K.py</code> evaluation across every scored BTC
      hourly market against the <code>{html.escape(segment_name)}</code> slice from
      <code>Model_K_Volatility_Decomposition_RT.py</code>. The volatility slice is defined as:
      {html.escape(segment_rule)}
    </p>
    <div class="hero-pills">
      <span class="pill"><span class="swatch vol"></span>{html.escape(segment_name)} hourly markets: {segment_bundle["n_hourly_markets"]:,}</span>
      <span class="pill"><span class="swatch total"></span>Total hourly markets: {total_hourly_markets:,}</span>
      <span class="pill">Segment share of markets: {format_number(segment_share, digits=4)}</span>
      <span class="pill">Outcome match rate: {format_number(segment_match_rate, digits=4)} vs {format_number(total_match_rate, digits=4)}</span>
    </div>
  </section>

  <h2>Topline Comparison</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(overview)}</div>
    <p class="chart-note">
      The total-market baseline includes the volatility segment itself. This dashboard is showing
      how the subset behaves relative to the full scored sample, not a disjoint holdout.
    </p>
  </section>

  <h2>Chart Comparison</h2>
  <section class="section-stack">
    <div class="chart-wrap">{overlay_chart_html}</div>
    <div class="chart-wrap">{brier_chart_html}</div>
  </section>

  <h2>Metric Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(metrics_comparison)}</div>
  </section>

  <h2>Brier Decomposition</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(decomposition_comparison)}</div>
  </section>

  <h2>Expanded Calibration Error</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(calibration_comparison)}</div>
  </section>

  <h2>Sharpness</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(sharpness_comparison)}</div>
  </section>

  <h2>Outcome Join Coverage</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(coverage_comparison)}</div>
  </section>

  <h2>Time Bucket Comparison</h2>
  <section class="section-stack">
    <section class="panel">
      <h3>Isolated Minute Metrics</h3>
      <div class="table-wrap">{render_table(isolated_metrics_comparison)}</div>
    </section>
    <section class="panel">
      <h3>10-Minute Bucket Metrics</h3>
      <div class="table-wrap">{render_table(decile_metrics_comparison)}</div>
    </section>
    <section class="panel">
      <h3>Accuracy Evaluation</h3>
      <div class="table-wrap">{render_table(time_bucket_accuracy_comparison)}</div>
    </section>
    <section class="panel">
      <h3>Brier Decomposition By Bucket</h3>
      <div class="table-wrap">{render_table(time_bucket_brier_comparison)}</div>
    </section>
  </section>

  <h2>Resolution Audit Mismatches</h2>
  <section class="section-stack">
    <section class="panel">
      <div class="table-wrap">{render_table(mismatch_count)}</div>
      <p class="chart-note">
        These audit mismatches are diagnostic only. They compare official Kalshi settlement results
        against the Binance strike audit already used by the existing scripts and are not scored as truth.
      </p>
    </section>
    <section class="panel">
      {side_by_side_table_html(segment_name, segment_mismatches, TOTAL_LABEL, total_mismatches)}
    </section>
  </section>

  <h2>Volatility Threshold Context</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(threshold_context_table(thresholds))}</div>
    <p class="chart-note">
      The real-time volatility segmentation uses a trailing <code>{ROLLING_WINDOW_HOURS}</code>-hour
      window, equivalent to <code>{ROLLING_WINDOW_MINUTES}</code> Binance one-minute observations.
    </p>
  </section>

  <h2>Assumptions</h2>
  <section class="panel">
    <ul>{assumption_html}</ul>
  </section>

  <h2>Run Notes</h2>
  <section class="panel">
    <ul>{notes_html}</ul>
    <p class="chart-note"><a href="../index.html">Back to dashboard index</a></p>
  </section>
</main>
</body>
</html>
"""


def build_index_html(
    *,
    thresholds: pd.DataFrame,
    summary_rows: pd.DataFrame,
    assumptions: Sequence[str],
    notes: Sequence[str],
) -> str:
    assumption_html = "".join(f"<li>{html.escape(item)}</li>" for item in assumptions)
    note_html = "".join(f"<li>{html.escape(item)}</li>" for item in notes)
    link_html = "".join(
        (
            "<li>"
            f"<a href='{html.escape(str(row['segment_name']) + '/comparison_dashboard.html')}'>"
            f"{html.escape(str(row['segment_name']))}</a>"
            f" ({html.escape(str(row['segment_rule']))})"
            "</li>"
        )
        for row in summary_rows.to_dict(orient="records")
    )

    summary_display = summary_rows.drop(columns=["segment_rule", "dashboard_path"]).copy()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model K Volatility Comparison Dashboards</title>
  {dashboard_style()}
</head>
<body>
<main>
  <section class="hero">
    <p class="eyebrow">Model K Dashboard Index</p>
    <h1>Total Market vs Volatility Segment Comparisons</h1>
    <p class="lead">
      This output keeps the same scoring pipeline from <code>Model_K.py</code> and
      <code>Model_K_Volatility_Decomposition_RT.py</code>, but reorganizes the results into
      comparison dashboards that line up each volatility slice against the total market baseline.
    </p>
    <div class="hero-pills">
      <span class="pill"><span class="swatch total"></span>{TOTAL_LABEL}</span>
      <span class="pill"><span class="swatch vol"></span>{VOLATILITY_LABEL}</span>
    </div>
  </section>

  <h2>Segment Dashboard Links</h2>
  <section class="panel">
    <ul class="links">{link_html}</ul>
  </section>

  <h2>Segment Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(summary_display)}</div>
  </section>

  <h2>Volatility Threshold Summary</h2>
  <section class="panel">
    <div class="table-wrap">{render_table(threshold_context_table(thresholds))}</div>
  </section>

  <h2>Shared Assumptions</h2>
  <section class="panel">
    <ul>{assumption_html}</ul>
  </section>

  <h2>Run Notes</h2>
  <section class="panel">
    <ul>{note_html}</ul>
  </section>
</main>
</body>
</html>
"""


def write_segment_artifacts(
    *,
    segment_dir: Path,
    segment_name: str,
    segment_rule: str,
    total_bundle: Dict[str, Any],
    segment_bundle: Dict[str, Any],
    total_hourly_markets: int,
    thresholds: pd.DataFrame,
    assumptions: Sequence[str],
    notes: Sequence[str],
) -> None:
    segment_dir.mkdir(parents=True, exist_ok=True)

    overview = overview_comparison_table(total_bundle, segment_bundle)
    metrics_comparison = build_comparison_table(
        total_df=total_bundle["metrics"],
        segment_df=segment_bundle["metrics"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "n_event_contracts",
            "base_rate",
            "mean_p_kalshi",
            "brier_score",
            "log_loss",
            "classification_accuracy",
        ],
    )
    decomposition_comparison = build_comparison_table(
        total_df=total_bundle["decomposition"],
        segment_df=segment_bundle["decomposition"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "brier_score",
            "reliability",
            "resolution",
            "uncertainty",
            "brier_from_decomposition",
        ],
    )
    calibration_comparison = build_comparison_table(
        total_df=total_bundle["expanded_summary"],
        segment_df=segment_bundle["expanded_summary"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "expected_calibration_error",
            "root_mean_squared_calibration_error",
            "max_calibration_error",
        ],
    )
    sharpness_comparison = build_comparison_table(
        total_df=total_bundle["sharpness"],
        segment_df=segment_bundle["sharpness"],
        key_cols=["segment"],
        value_cols=[
            "n_forecasts",
            "base_rate",
            "mean_p_kalshi",
            "forecast_std",
            "sharpness_variance_from_base_rate",
            "mean_abs_distance_from_0_5",
            "mean_predictive_variance_p_times_1_minus_p",
        ],
    )
    coverage_comparison = build_comparison_table(
        total_df=total_bundle["coverage"],
        segment_df=segment_bundle["coverage"],
        key_cols=["scope", "value"],
        value_cols=["total_forecast_rows", "matched_rows", "unmatched_rows", "match_rate"],
    )
    isolated_metrics_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_metrics"][
            (total_bundle["time_bucket_metrics"]["bucket_type"] == "isolated")
            & (total_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        segment_df=segment_bundle["time_bucket_metrics"][
            (segment_bundle["time_bucket_metrics"]["bucket_type"] == "isolated")
            & (segment_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "n_event_contracts",
            "base_rate",
            "mean_p_kalshi",
            "brier_score",
            "log_loss",
            "classification_accuracy",
        ],
    )
    decile_metrics_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_metrics"][
            (total_bundle["time_bucket_metrics"]["bucket_type"] == "decile")
            & (total_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        segment_df=segment_bundle["time_bucket_metrics"][
            (segment_bundle["time_bucket_metrics"]["bucket_type"] == "decile")
            & (segment_bundle["time_bucket_metrics"]["n_forecasts"] > 0)
        ],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "n_event_contracts",
            "base_rate",
            "mean_p_kalshi",
            "brier_score",
            "log_loss",
            "classification_accuracy",
        ],
    )
    time_bucket_accuracy_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_accuracy"][total_bundle["time_bucket_accuracy"]["n_forecasts"] > 0],
        segment_df=segment_bundle["time_bucket_accuracy"][segment_bundle["time_bucket_accuracy"]["n_forecasts"] > 0],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "correct_forecasts",
            "incorrect_forecasts",
            "classification_accuracy",
            "classification_accuracy_pct",
            "threshold",
        ],
    )
    time_bucket_brier_comparison = build_comparison_table(
        total_df=total_bundle["time_bucket_brier"][total_bundle["time_bucket_brier"]["n_forecasts"] > 0],
        segment_df=segment_bundle["time_bucket_brier"][segment_bundle["time_bucket_brier"]["n_forecasts"] > 0],
        key_cols=["sort_order", "display_name"],
        hidden_key_cols=["sort_order"],
        sort_cols=["sort_order"],
        value_cols=[
            "n_forecasts",
            "brier_score",
            "reliability",
            "resolution",
            "uncertainty",
            "brier_from_decomposition",
        ],
    )

    overview.to_csv(segment_dir / "overview_comparison.csv", index=False)
    metrics_comparison.to_csv(segment_dir / "metrics_summary_comparison.csv", index=False)
    decomposition_comparison.to_csv(segment_dir / "brier_decomposition_comparison.csv", index=False)
    calibration_comparison.to_csv(segment_dir / "expanded_calibration_error_summary_comparison.csv", index=False)
    sharpness_comparison.to_csv(segment_dir / "sharpness_comparison.csv", index=False)
    coverage_comparison.to_csv(segment_dir / "outcome_join_coverage_comparison.csv", index=False)
    isolated_metrics_comparison.to_csv(segment_dir / "isolated_minute_metrics_comparison.csv", index=False)
    decile_metrics_comparison.to_csv(segment_dir / "ten_minute_bucket_metrics_comparison.csv", index=False)
    time_bucket_accuracy_comparison.to_csv(segment_dir / "time_bucket_accuracy_comparison.csv", index=False)
    time_bucket_brier_comparison.to_csv(segment_dir / "time_bucket_brier_decomposition_comparison.csv", index=False)
    segment_bundle["resolution_mismatches"].to_csv(segment_dir / "segment_resolution_mismatches.csv", index=False)
    total_bundle["resolution_mismatches"].to_csv(segment_dir / "total_resolution_mismatches.csv", index=False)

    html_document = build_segment_dashboard_html(
        segment_name=segment_name,
        segment_rule=segment_rule,
        total_bundle=total_bundle,
        segment_bundle=segment_bundle,
        total_hourly_markets=total_hourly_markets,
        thresholds=thresholds,
        assumptions=assumptions,
        notes=notes,
    )
    (segment_dir / "comparison_dashboard.html").write_text(html_document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root_from_script()
    output_root = resolve_path(root, args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    forecasts = load_kalshi_price_outputs(resolve_path(root, args.kalshi_price_dir))
    settlements = load_settlements(resolve_path(root, args.settlement_csv))
    outcomes, source_notes = build_kalshi_reality_outcomes(
        forecasts=forecasts,
        settlements=settlements,
        output_dir=output_root,
    )
    raw, unmatched = attach_outcomes(forecasts, outcomes)

    hourly_market_states, thresholds, state_notes = compute_or_load_hourly_market_states(
        raw=raw,
        output_root=output_root,
        root=root,
        hourly_market_states_csv=args.hourly_market_states_csv,
        binance_minute_cache_csv=args.binance_minute_cache_csv,
        refresh_binance_cache=args.refresh_binance_cache,
    )
    hourly_market_states = hourly_market_states.sort_values("forecast_hour_start_utc").drop_duplicates(
        subset=["event_ticker"],
        keep="last",
    )
    thresholds.to_csv(output_root / "segment_thresholds.csv", index=False)
    hourly_market_states.to_csv(output_root / "hourly_market_volatility_segments.csv", index=False)

    total_hourly_markets = int(hourly_market_states["event_ticker"].nunique())
    total_bundle = build_bundle(
        name=TOTAL_LABEL,
        raw=raw,
        forecasts=forecasts,
        unmatched=unmatched,
        calibration_bins=args.calibration_bins,
        classification_threshold=args.classification_threshold,
        skip_binance_audit=args.skip_binance_audit,
        n_hourly_markets=total_hourly_markets,
    )

    combined_assumptions = list(ASSUMPTIONS) + list(volatility_assumptions(thresholds))
    shared_notes = [
        *source_notes,
        *state_notes,
        "Dashboard inputs come from Data_Sourcing/Kalshi_Pricing_Fetch and Data_Sourcing/Settlement_Outcomes, matching the existing scripts.",
        "This formatter keeps the same Model K scoring, calibration, Brier decomposition, and time-bucket logic. Only the presentation layer is new.",
    ]

    summary_rows: List[Dict[str, Any]] = []
    for segment in SEGMENTS:
        event_ids = set(hourly_market_states.loc[hourly_market_states[segment["flag"]], "event_ticker"].astype(str))
        segment_forecasts = slice_by_event_ids(forecasts, event_ids)
        segment_raw = slice_by_event_ids(raw, event_ids)
        segment_unmatched = slice_by_event_ids(unmatched, event_ids)

        segment_bundle = build_bundle(
            name=segment["name"],
            raw=segment_raw,
            forecasts=segment_forecasts,
            unmatched=segment_unmatched,
            calibration_bins=args.calibration_bins,
            classification_threshold=args.classification_threshold,
            skip_binance_audit=args.skip_binance_audit,
            n_hourly_markets=len(event_ids),
        )

        segment_dir = output_root / segment["name"]
        write_segment_artifacts(
            segment_dir=segment_dir,
            segment_name=segment["name"],
            segment_rule=segment["rule"],
            total_bundle=total_bundle,
            segment_bundle=segment_bundle,
            total_hourly_markets=total_hourly_markets,
            thresholds=thresholds,
            assumptions=combined_assumptions,
            notes=shared_notes,
        )

        total_overall = total_bundle["overall_metrics"]
        segment_overall = segment_bundle["overall_metrics"]
        total_ece = total_bundle["overall_ece"]
        segment_ece = segment_bundle["overall_ece"]
        summary_rows.append(
            {
                "segment_name": segment["name"],
                "segment_rule": segment["rule"],
                "n_hourly_markets": len(event_ids),
                "hourly_market_share": (len(event_ids) / total_hourly_markets) if total_hourly_markets else np.nan,
                "scored_rows_volatility": segment_overall["n_forecasts"],
                "scored_rows_total": total_overall["n_forecasts"],
                "brier_score_volatility": segment_overall["brier_score"],
                "brier_score_total": total_overall["brier_score"],
                "log_loss_volatility": segment_overall["log_loss"],
                "log_loss_total": total_overall["log_loss"],
                "ece_volatility": segment_ece["expected_calibration_error"],
                "ece_total": total_ece["expected_calibration_error"],
                "audit_mismatches_volatility": len(segment_bundle["resolution_mismatches"]),
                "audit_mismatches_total": len(total_bundle["resolution_mismatches"]),
                "dashboard_path": f"{segment['name']}/comparison_dashboard.html",
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_root / "segment_dashboard_summary.csv", index=False)

    index_html = build_index_html(
        thresholds=thresholds,
        summary_rows=summary_df,
        assumptions=combined_assumptions,
        notes=shared_notes,
    )
    (output_root / "index.html").write_text(index_html, encoding="utf-8")

    print(f"Comparison dashboards written to: {output_root}")
    print(f"Index dashboard: {output_root / 'index.html'}")


if __name__ == "__main__":
    main()
