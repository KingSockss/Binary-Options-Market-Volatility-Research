from __future__ import annotations

import argparse
import html
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError as exc:  # pragma: no cover - Python 3.9+ is expected.
    raise SystemExit("Python 3.9+ required for zoneinfo.") from exc

try:  # Plotly is preferred, but the repository venv may not have it installed.
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except Exception:  # pragma: no cover - exercised when Plotly is unavailable.
    go = None
    make_subplots = None
    HAS_PLOTLY = False


NY = ZoneInfo("America/New_York")
UTC = timezone.utc

SYMBOL = "BTCUSDT"

CONTRACT_LABELS = ("ATM", "OTM+1", "OTM-1")
EPS = 1e-15
ASSUMPTIONS = (
    "Kalshi prices are YES probabilities. CSV values greater than 1 and no greater than 100 "
    "are treated as cents and divided by 100 before scoring.",
    "Scored event outcome is the official Kalshi contract result: 1 for YES resolved, 0 for NO resolved.",
    "Forecast rows are joined to outcomes by exact Kalshi market ticker when available; legacy price files "
    "without market tickers fall back to event_ticker + contract_label + strike.",
    "Event tickers are interpreted as the contract settlement/end hour, matching the price fetcher's "
    "hour_end_utc -> New York event ticker convention.",
    "Binance prices are diagnostic only. They may be used for strike matching upstream and audit mismatch checks, "
    "but never as the scored truth label.",
    "Events present in the Kalshi price files but absent from the settlement CSV are excluded "
    "from scoring and counted in outcome_join_coverage.csv.",
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_settlement_csv(root: Path) -> Path:
    corrected = root / "Data_Sourcing" / "Settlement_Outcomes" / "kalshi_btc_atm_settlements.csv"
    typo_legacy = root / "Data_Sourcing" / "Settlement_Outocmes" / "kalshi_btc_atm_settlements.csv"
    return corrected if corrected.exists() or not typo_legacy.exists() else typo_legacy


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Kalshi BTC hourly contract probabilities against realized "
            "Binance settlement outcomes."
        )
    )
    parser.add_argument(
        "--kalshi-price-dir",
        type=Path,
        default=root / "Data_Sourcing" / "Kalshi_Pricing_Fetch" / "hourly_events_price_data",
        help="Folder containing Kalshi_Contract_Price_Fetch.py hourly CSV outputs.",
    )
    parser.add_argument(
        "--settlement-csv",
        type=Path,
        default=default_settlement_csv(root),
        help="Settlement CSV with official Kalshi result/outcome columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Model_K_outputs",
        help="Directory for Model K output files.",
    )
    parser.add_argument(
        "--skip-binance-audit",
        action="store_true",
        help="Skip diagnostic comparison between official Kalshi outcomes and Binance audit prices.",
    )
    parser.add_argument("--calibration-bins", type=int, default=10, help="Number of calibration bins.")
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Threshold used for binary classification accuracy in the metric summary.",
    )
    return parser.parse_args()


def event_time_from_ticker(event_ticker: str) -> pd.Timestamp:
    match = re.fullmatch(r"KXBTCD-(\d{2})([A-Z]{3})(\d{2})(\d{2})", str(event_ticker))
    if not match:
        raise ValueError(f"Cannot parse Kalshi BTC event ticker: {event_ticker}")

    yy, mon_text, dd, hh = match.groups()
    month = datetime.strptime(mon_text, "%b").month
    dt_ny = datetime(2000 + int(yy), month, int(dd), int(hh), tzinfo=NY)
    return pd.Timestamp(dt_ny.astimezone(UTC))


def normalize_kalshi_probability(series: pd.Series) -> Tuple[pd.Series, str]:
    numeric = pd.to_numeric(series, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        return numeric, "empty"

    max_value = float(non_null.max())
    min_value = float(non_null.min())
    if max_value <= 1.0 and min_value >= 0.0:
        return numeric, "probability"
    if max_value <= 100.0 and min_value >= 0.0:
        return numeric / 100.0, "cents_divided_by_100"
    raise ValueError(
        "Kalshi price values must be probabilities in [0, 1] or cents in [0, 100]. "
        f"Observed min={min_value}, max={max_value}."
    )


def load_kalshi_price_outputs(price_dir: Path) -> pd.DataFrame:
    files = sorted(price_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No Kalshi price CSV files found in {price_dir}")

    frames: List[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path)
        required = {"Event", "datetime", "minute_number"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        df["source_file"] = path.name
        frames.append(df)

    wide = pd.concat(frames, ignore_index=True)
    wide["forecast_datetime_utc"] = pd.to_datetime(wide["datetime"], utc=True)

    rows: List[pd.DataFrame] = []
    for label in CONTRACT_LABELS:
        strike_col = f"{label}_strike"
        price_col = f"{label}_price"
        market_col = f"{label}_market_ticker"
        if strike_col not in wide.columns or price_col not in wide.columns:
            continue
        source_cols = [
            "Event",
            "forecast_datetime_utc",
            "minute_number",
            "source_file",
            strike_col,
            price_col,
        ]
        if market_col in wide.columns:
            source_cols.append(market_col)
        part = wide[
            source_cols
        ].copy()
        rename_cols = {
            "Event": "event_ticker",
            strike_col: "strike",
            price_col: "p_kalshi_raw",
        }
        if market_col in part.columns:
            rename_cols[market_col] = "market_ticker"
        part = part.rename(columns=rename_cols)
        if "market_ticker" not in part.columns:
            part["market_ticker"] = pd.NA
        part["contract_label"] = label
        part["p_kalshi"], scale = normalize_kalshi_probability(part["p_kalshi_raw"])
        part["kalshi_price_scale"] = scale
        rows.append(part)

    if not rows:
        raise ValueError(f"No contract price columns found in {price_dir}")

    out = pd.concat(rows, ignore_index=True)
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    out["p_kalshi_raw"] = pd.to_numeric(out["p_kalshi_raw"], errors="coerce")
    out["market_ticker"] = out["market_ticker"].astype("string").str.strip()
    out.loc[out["market_ticker"].isin(["", "nan", "None", "<NA>"]), "market_ticker"] = pd.NA
    out = out.dropna(subset=["event_ticker", "forecast_datetime_utc", "strike", "p_kalshi"])
    out["p_kalshi"] = out["p_kalshi"].clip(0.0, 1.0)
    out["event_contract_id"] = (
        out["event_ticker"].astype(str) + "|" + out["contract_label"].astype(str) + "|" + out["strike"].round(8).astype(str)
    )
    out["forecast_row_id"] = np.arange(len(out), dtype=int)
    return out.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label"]).reset_index(drop=True)


def normalize_result_to_outcome(value: Any) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"yes", "y", "true", "1", "win", "won"}:
        return 1
    if text in {"no", "n", "false", "0", "lose", "lost"}:
        return 0
    return None


def load_settlements(settlement_csv: Path) -> pd.DataFrame:
    if not settlement_csv.exists():
        raise FileNotFoundError(f"Settlement CSV not found: {settlement_csv}")

    df = pd.read_csv(settlement_csv)
    required = {"event_ticker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{settlement_csv} is missing columns: {sorted(missing)}")

    if "event_datetime" in df.columns:
        df["event_datetime_utc"] = pd.to_datetime(df["event_datetime"], utc=True)
    elif "datetime" in df.columns:
        df["event_datetime_utc"] = pd.to_datetime(df["datetime"], utc=True)
    else:
        df["event_datetime_utc"] = df["event_ticker"].map(event_time_from_ticker)

    parsed = df["event_ticker"].map(event_time_from_ticker)
    df["event_datetime_utc"] = df["event_datetime_utc"].fillna(parsed)
    return df.dropna(subset=["event_ticker", "event_datetime_utc"]).copy()


def settlement_outcomes_long(settlements: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for label in ("ATM", "OTM+1", "OTM-1", "OTM+2", "OTM-2"):
        market_col = f"{label}_market_ticker"
        strike_col = f"{label}_strike"
        result_col = f"{label}_result"
        legacy_result_col = f"{label}_settlement"
        outcome_col = f"{label}_outcome"

        if market_col not in settlements.columns or strike_col not in settlements.columns:
            continue
        if outcome_col not in settlements.columns and result_col not in settlements.columns and legacy_result_col not in settlements.columns:
            continue

        source_cols = ["event_ticker", "event_datetime_utc", market_col, strike_col]
        for optional_col in [
            result_col,
            legacy_result_col,
            outcome_col,
            "binance_audit_price",
            "binance_reference_price",
            "forecast_hour_start_datetime",
        ]:
            if optional_col in settlements.columns:
                source_cols.append(optional_col)
        if "binance_audit_price" not in settlements.columns and "binance_price" in settlements.columns:
            source_cols.append("binance_price")

        part = settlements[source_cols].copy()
        rename_cols = {
            market_col: "market_ticker",
            strike_col: "strike",
        }
        if result_col in part.columns:
            rename_cols[result_col] = "official_result"
        elif legacy_result_col in part.columns:
            rename_cols[legacy_result_col] = "official_result"
        if outcome_col in part.columns:
            rename_cols[outcome_col] = "p_reality"
        if "binance_price" in part.columns:
            rename_cols["binance_price"] = "binance_audit_price"
        part = part.rename(columns=rename_cols)
        part["contract_label"] = label
        rows.append(part)

    if not rows:
        raise ValueError("Settlement CSV does not contain recognizable Kalshi outcome columns.")

    out = pd.concat(rows, ignore_index=True)
    out["market_ticker"] = out["market_ticker"].astype("string").str.strip()
    out.loc[out["market_ticker"].isin(["", "nan", "None", "<NA>"]), "market_ticker"] = pd.NA
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    if "p_reality" in out.columns:
        out["p_reality"] = pd.to_numeric(out["p_reality"], errors="coerce")
    else:
        out["p_reality"] = np.nan
    if "official_result" not in out.columns:
        out["official_result"] = pd.NA
    missing_outcome = out["p_reality"].isna()
    if missing_outcome.any():
        result_outcomes = out.loc[missing_outcome, "official_result"].map(normalize_result_to_outcome)
        out.loc[missing_outcome, "p_reality"] = pd.to_numeric(result_outcomes, errors="coerce")
    if "binance_audit_price" in out.columns:
        out["binance_audit_price"] = pd.to_numeric(out["binance_audit_price"], errors="coerce")
    else:
        out["binance_audit_price"] = np.nan
    if "binance_reference_price" in out.columns:
        out["binance_reference_price"] = pd.to_numeric(out["binance_reference_price"], errors="coerce")
    else:
        out["binance_reference_price"] = np.nan

    out["p_reality"] = pd.to_numeric(out["p_reality"], errors="coerce")
    out = out.dropna(subset=["event_ticker", "strike", "p_reality"]).copy()
    out["p_reality"] = out["p_reality"].astype(int).clip(0, 1)
    out["strike_key"] = out["strike"].round(8)
    out["settlement_source"] = "kalshi_official"
    return out.drop_duplicates(
        subset=["event_ticker", "market_ticker", "contract_label", "strike_key"],
        keep="first",
    ).reset_index(drop=True)


def build_kalshi_reality_outcomes(
    forecasts: pd.DataFrame,
    settlements: pd.DataFrame,
    output_dir: Path,
) -> Tuple[pd.DataFrame, List[str]]:
    messages: List[str] = []
    outcomes = settlement_outcomes_long(settlements)
    outcomes.to_csv(output_dir / "kalshi_official_outcomes_long.csv", index=False)
    messages.append("Loaded official Kalshi contract outcomes from kalshi_btc_atm_settlements.csv.")
    return outcomes, messages


def attach_outcomes(forecasts: pd.DataFrame, outcomes: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = forecasts.copy()
    forecasts["strike_key"] = forecasts["strike"].round(8)
    outcomes = outcomes.copy()
    outcomes["strike_key"] = outcomes["strike"].round(8)

    primary_forecasts = forecasts[forecasts["market_ticker"].notna()].copy()
    legacy_forecasts = forecasts[forecasts["market_ticker"].isna()].copy()

    outcome_cols = [
        "event_ticker",
        "event_datetime_utc",
        "market_ticker",
        "contract_label",
        "strike_key",
        "official_result",
        "p_reality",
        "settlement_source",
        "binance_audit_price",
        "binance_reference_price",
    ]
    outcome_cols = [col for col in outcome_cols if col in outcomes.columns]

    matched_parts: List[pd.DataFrame] = []
    if not primary_forecasts.empty:
        primary = primary_forecasts.merge(
            outcomes[outcome_cols],
            on=["event_ticker", "market_ticker"],
            how="left",
            suffixes=("", "_settlement"),
        )
        primary["join_key_used"] = "event_ticker+market_ticker"
        matched_parts.append(primary)

    if not legacy_forecasts.empty:
        fallback = legacy_forecasts.merge(
            outcomes[outcome_cols],
            on=["event_ticker", "contract_label", "strike_key"],
            how="left",
            suffixes=("", "_settlement"),
        )
        fallback["join_key_used"] = "event_ticker+contract_label+strike"
        matched_parts.append(fallback)

    df = pd.concat(matched_parts, ignore_index=True) if matched_parts else pd.DataFrame()
    unmatched = df[df["p_reality"].isna()].copy()
    df = df.dropna(subset=["p_reality"]).copy()
    if df.empty:
        raise ValueError("No forecast rows matched official Kalshi settlement outcomes.")

    if "strike_settlement" in df.columns:
        df["settlement_strike"] = df["strike_settlement"]
        df = df.drop(columns=["strike_settlement"])
    if "contract_label_settlement" in df.columns:
        df = df.drop(columns=["contract_label_settlement"])

    df["p_reality"] = df["p_reality"].astype(int)
    df["outcome"] = df["p_reality"].astype(int)
    df["forecast_error"] = df["p_kalshi"] - df["p_reality"]
    df["squared_error"] = df["forecast_error"] ** 2
    df["absolute_error"] = df["forecast_error"].abs()
    df["log_loss_component"] = -(
        df["p_reality"] * np.log(df["p_kalshi"].clip(EPS, 1 - EPS))
        + (1 - df["p_reality"]) * np.log((1 - df["p_kalshi"]).clip(EPS, 1 - EPS))
    )
    df["minutes_to_settlement"] = (
        (df["event_datetime_utc"] - df["forecast_datetime_utc"]).dt.total_seconds() / 60.0
    )
    return (
        df.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label"]).reset_index(drop=True),
        unmatched.sort_values(["forecast_datetime_utc", "event_ticker", "contract_label"]).reset_index(drop=True),
    )


def build_outcome_join_coverage(forecasts: pd.DataFrame, scored: pd.DataFrame, unmatched: pd.DataFrame) -> pd.DataFrame:
    matched_ids = set(scored["forecast_row_id"].tolist())
    work = forecasts.copy()
    work["matched_official_outcome"] = work["forecast_row_id"].isin(matched_ids)

    rows: List[Dict[str, Any]] = []

    def add_row(scope: str, value: str, frame: pd.DataFrame) -> None:
        matched = int(frame["matched_official_outcome"].sum())
        total = int(len(frame))
        rows.append(
            {
                "scope": scope,
                "value": value,
                "total_forecast_rows": total,
                "matched_rows": matched,
                "unmatched_rows": total - matched,
                "match_rate": matched / total if total else np.nan,
            }
        )

    add_row("overall", "all", work)
    for label, part in work.groupby("contract_label", sort=True):
        add_row("contract_label", str(label), part)
    for source_file, part in work.groupby("source_file", sort=True):
        add_row("source_file", str(source_file), part)

    return pd.DataFrame(rows)


def build_resolution_mismatches(scored: pd.DataFrame) -> pd.DataFrame:
    if "binance_audit_price" not in scored.columns:
        return pd.DataFrame()

    audit = scored.dropna(subset=["binance_audit_price", "strike"]).copy()
    if audit.empty:
        return pd.DataFrame()

    audit["binance_audit_outcome"] = (audit["binance_audit_price"] >= audit["strike"]).astype(int)
    mismatches = audit[audit["binance_audit_outcome"] != audit["outcome"]].copy()
    cols = [
        "event_ticker",
        "event_datetime_utc",
        "contract_label",
        "market_ticker",
        "strike",
        "official_result",
        "outcome",
        "binance_audit_price",
        "binance_audit_outcome",
        "source_file",
    ]
    return mismatches[[col for col in cols if col in mismatches.columns]].drop_duplicates().reset_index(drop=True)


def metric_row(name: str, frame: pd.DataFrame, threshold: float) -> Dict[str, Any]:
    if frame.empty:
        return {
            "segment": name,
            "n_forecasts": 0,
            "n_event_contracts": 0,
            "base_rate": np.nan,
            "mean_p_kalshi": np.nan,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "classification_accuracy": np.nan,
        }

    predicted = (frame["p_kalshi"] >= threshold).astype(int)
    return {
        "segment": name,
        "n_forecasts": int(len(frame)),
        "n_event_contracts": int(frame["event_contract_id"].nunique()),
        "base_rate": float(frame["p_reality"].mean()),
        "mean_p_kalshi": float(frame["p_kalshi"].mean()),
        "brier_score": float(frame["squared_error"].mean()),
        "log_loss": float(frame["log_loss_component"].mean()),
        "classification_accuracy": float((predicted == frame["outcome"]).mean()),
    }


def build_metrics_summary(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = [metric_row("overall", df, threshold)]
    for label, part in df.groupby("contract_label", sort=True):
        rows.append(metric_row(str(label), part, threshold))
    return pd.DataFrame(rows)


def calibration_table(df: pd.DataFrame, bins: int, segment: str = "overall") -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(bins)]
    work = df.copy()
    work["calibration_bin"] = pd.cut(
        work["p_kalshi"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    grouped = work.groupby("calibration_bin", observed=False)
    out = grouped.agg(
        n_forecasts=("p_kalshi", "size"),
        mean_p_kalshi=("p_kalshi", "mean"),
        observed_frequency=("p_reality", "mean"),
    ).reset_index()
    out["segment"] = segment
    out["bin_lower"] = edges[:-1]
    out["bin_upper"] = edges[1:]
    total = max(int(out["n_forecasts"].sum()), 1)
    out["forecast_share"] = out["n_forecasts"] / total
    out["signed_calibration_error"] = out["mean_p_kalshi"] - out["observed_frequency"]
    out["absolute_calibration_error"] = out["signed_calibration_error"].abs()
    out["weighted_absolute_calibration_error"] = out["forecast_share"] * out["absolute_calibration_error"]
    out["weighted_squared_calibration_error"] = out["forecast_share"] * (out["signed_calibration_error"] ** 2)
    return out[
        [
            "segment",
            "calibration_bin",
            "bin_lower",
            "bin_upper",
            "n_forecasts",
            "forecast_share",
            "mean_p_kalshi",
            "observed_frequency",
            "signed_calibration_error",
            "absolute_calibration_error",
            "weighted_absolute_calibration_error",
            "weighted_squared_calibration_error",
        ]
    ]


def brier_decomposition(df: pd.DataFrame, bins: int, segment: str = "overall") -> Dict[str, Any]:
    cal = calibration_table(df, bins=bins, segment=segment)
    nonempty = cal[cal["n_forecasts"] > 0].copy()
    if df.empty or nonempty.empty:
        return {
            "segment": segment,
            "n_forecasts": 0,
            "brier_score": np.nan,
            "reliability": np.nan,
            "resolution": np.nan,
            "uncertainty": np.nan,
            "brier_from_decomposition": np.nan,
        }

    base_rate = float(df["p_reality"].mean())
    reliability = float(
        (
            nonempty["forecast_share"]
            * (nonempty["mean_p_kalshi"] - nonempty["observed_frequency"]) ** 2
        ).sum()
    )
    resolution = float((nonempty["forecast_share"] * (nonempty["observed_frequency"] - base_rate) ** 2).sum())
    uncertainty = float(base_rate * (1.0 - base_rate))
    return {
        "segment": segment,
        "n_forecasts": int(len(df)),
        "brier_score": float(df["squared_error"].mean()),
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier_from_decomposition": reliability - resolution + uncertainty,
    }


def build_brier_decomposition(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows = [brier_decomposition(df, bins=bins, segment="overall")]
    for label, part in df.groupby("contract_label", sort=True):
        rows.append(brier_decomposition(part, bins=bins, segment=str(label)))
    return pd.DataFrame(rows)


def expanded_calibration_error(df: pd.DataFrame, bins: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = [calibration_table(df, bins=bins, segment="overall")]
    for label, part in df.groupby("contract_label", sort=True):
        rows.append(calibration_table(part, bins=bins, segment=str(label)))
    expanded = pd.concat(rows, ignore_index=True)

    summary_rows: List[Dict[str, Any]] = []
    for segment, part in expanded.groupby("segment", sort=False):
        nonempty = part[part["n_forecasts"] > 0]
        summary_rows.append(
            {
                "segment": segment,
                "n_forecasts": int(nonempty["n_forecasts"].sum()),
                "expected_calibration_error": float(nonempty["weighted_absolute_calibration_error"].sum()),
                "root_mean_squared_calibration_error": float(
                    math.sqrt(nonempty["weighted_squared_calibration_error"].sum())
                ),
                "max_calibration_error": float(nonempty["absolute_calibration_error"].max())
                if not nonempty.empty
                else np.nan,
            }
        )
    return expanded, pd.DataFrame(summary_rows)


def build_sharpness(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for segment, part in [("overall", df), *[(str(k), v) for k, v in df.groupby("contract_label", sort=True)]]:
        base_rate = float(part["p_reality"].mean()) if not part.empty else np.nan
        rows.append(
            {
                "segment": segment,
                "n_forecasts": int(len(part)),
                "base_rate": base_rate,
                "mean_p_kalshi": float(part["p_kalshi"].mean()) if not part.empty else np.nan,
                "forecast_std": float(part["p_kalshi"].std(ddof=0)) if not part.empty else np.nan,
                "sharpness_variance_from_base_rate": float(((part["p_kalshi"] - base_rate) ** 2).mean())
                if not part.empty
                else np.nan,
                "mean_abs_distance_from_0_5": float((part["p_kalshi"] - 0.5).abs().mean())
                if not part.empty
                else np.nan,
                "mean_predictive_variance_p_times_1_minus_p": float((part["p_kalshi"] * (1 - part["p_kalshi"])).mean())
                if not part.empty
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_market_elapsed_minutes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["market_minute"] = (pd.to_numeric(out["minute_number"], errors="coerce") + 1).round().astype("Int64")
    return out


def time_bucket_specs(max_minute: int = 60) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = [
        {
            "bucket_type": "isolated",
            "bucket": "minute_1",
            "display_name": "Minute 1",
            "lower_minute": 1,
            "upper_minute": 1,
            "sort_order": 1,
        },
        {
            "bucket_type": "isolated",
            "bucket": "minute_30",
            "display_name": "Minute 30",
            "lower_minute": 30,
            "upper_minute": 30,
            "sort_order": 2,
        },
    ]

    sort_order = 100
    for lower in range(1, max_minute + 1, 10):
        upper = min(lower + 9, max_minute)
        specs.append(
            {
                "bucket_type": "decile",
                "bucket": f"minutes_{lower}_{upper}",
                "display_name": f"Minutes {lower}-{upper}",
                "lower_minute": lower,
                "upper_minute": upper,
                "sort_order": sort_order,
            }
        )
        sort_order += 1
    return specs


def time_bucket_frame(df: pd.DataFrame, spec: Dict[str, Any]) -> pd.DataFrame:
    minutes = df["market_minute"]
    return df[(minutes >= spec["lower_minute"]) & (minutes <= spec["upper_minute"])].copy()


def build_time_bucket_outputs(
    df: pd.DataFrame,
    *,
    calibration_bins: int,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = add_market_elapsed_minutes(df)
    max_minute_value = work["market_minute"].dropna().max()
    max_minute = int(max(60, max_minute_value)) if pd.notna(max_minute_value) else 60

    metric_rows: List[Dict[str, Any]] = []
    accuracy_rows: List[Dict[str, Any]] = []
    brier_rows: List[Dict[str, Any]] = []
    calibration_rows: List[pd.DataFrame] = []

    for spec in time_bucket_specs(max_minute=max_minute):
        part = time_bucket_frame(work, spec)
        segment = spec["display_name"]

        metric = metric_row(segment, part, threshold)
        brier = brier_decomposition(part, bins=calibration_bins, segment=segment)
        predicted = (part["p_kalshi"] >= threshold).astype(int) if not part.empty else pd.Series(dtype=int)
        correct = int((predicted == part["outcome"]).sum()) if not part.empty else 0

        metadata = {
            "bucket_type": spec["bucket_type"],
            "bucket": spec["bucket"],
            "display_name": spec["display_name"],
            "lower_minute": spec["lower_minute"],
            "upper_minute": spec["upper_minute"],
            "sort_order": spec["sort_order"],
        }
        metric_rows.append({**metadata, **metric})
        brier_rows.append({**metadata, **brier})
        accuracy_rows.append(
            {
                **metadata,
                "n_forecasts": int(len(part)),
                "correct_forecasts": correct,
                "incorrect_forecasts": int(len(part) - correct),
                "classification_accuracy": float(correct / len(part)) if len(part) else np.nan,
                "classification_accuracy_pct": float((correct / len(part)) * 100.0) if len(part) else np.nan,
                "threshold": threshold,
            }
        )

        cal = calibration_table(part, bins=calibration_bins, segment=segment)
        for key, value in metadata.items():
            cal[key] = value
        calibration_rows.append(cal)

    metrics = pd.DataFrame(metric_rows).sort_values(["sort_order"]).reset_index(drop=True)
    accuracy = pd.DataFrame(accuracy_rows).sort_values(["sort_order"]).reset_index(drop=True)
    brier = pd.DataFrame(brier_rows).sort_values(["sort_order"]).reset_index(drop=True)
    calibration = pd.concat(calibration_rows, ignore_index=True).sort_values(
        ["sort_order", "bin_lower"]
    ).reset_index(drop=True)
    return metrics, accuracy, brier, calibration


def write_individual_metric_files(
    *,
    output_dir: Path,
    raw: pd.DataFrame,
    metrics: pd.DataFrame,
    decomposition: pd.DataFrame,
    calibration: pd.DataFrame,
    expanded_error: pd.DataFrame,
    expanded_error_summary: pd.DataFrame,
    sharpness: pd.DataFrame,
    coverage: pd.DataFrame,
    unmatched: pd.DataFrame,
    resolution_mismatches: pd.DataFrame,
    time_bucket_metrics: pd.DataFrame,
    time_bucket_accuracy: pd.DataFrame,
    time_bucket_brier: pd.DataFrame,
    time_bucket_calibration: pd.DataFrame,
) -> None:
    raw.to_csv(output_dir / "raw_values.csv", index=False)
    coverage.to_csv(output_dir / "outcome_join_coverage.csv", index=False)
    unmatched.to_csv(output_dir / "unmatched_forecast_rows.csv", index=False)
    resolution_mismatches.to_csv(output_dir / "resolution_mismatches.csv", index=False)
    time_bucket_metrics.to_csv(output_dir / "time_bucket_metrics.csv", index=False)
    time_bucket_accuracy.to_csv(output_dir / "time_bucket_accuracy.csv", index=False)
    time_bucket_brier.to_csv(output_dir / "time_bucket_brier_decomposition.csv", index=False)
    time_bucket_calibration.to_csv(output_dir / "time_bucket_calibration_curve.csv", index=False)
    metrics[["segment", "n_forecasts", "n_event_contracts", "brier_score"]].to_csv(
        output_dir / "brier_score.csv", index=False
    )
    decomposition.to_csv(output_dir / "brier_decomposition.csv", index=False)
    metrics[["segment", "n_forecasts", "n_event_contracts", "log_loss"]].to_csv(
        output_dir / "log_loss.csv", index=False
    )
    calibration.to_csv(output_dir / "calibration_curve.csv", index=False)
    expanded_error.to_csv(output_dir / "expanded_calibration_error.csv", index=False)
    expanded_error_summary.to_csv(output_dir / "expanded_calibration_error_summary.csv", index=False)
    sharpness.to_csv(output_dir / "sharpness.csv", index=False)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "model_k_assumptions.txt").write_text("\n".join(ASSUMPTIONS) + "\n", encoding="utf-8")

    stale_removed_metric = output_dir / "directional_accuracy.csv"
    if stale_removed_metric.exists():
        stale_removed_metric.unlink()
    stale_binance_truth = output_dir / "binance_settlement_prices.csv"
    if stale_binance_truth.exists():
        stale_binance_truth.unlink()
    stale_binance_download = output_dir / "binance_1m_klines.csv"
    if stale_binance_download.exists():
        stale_binance_download.unlink()
    stale_excluded_events = output_dir / "excluded_events_missing_settlement.csv"
    if stale_excluded_events.exists():
        stale_excluded_events.unlink()


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return f"{float(value):,.{digits}f}"


def dataframe_to_html_table(df: pd.DataFrame, max_rows: Optional[int] = None) -> str:
    show = df if max_rows is None else df.head(max_rows)
    return show.to_html(index=False, border=0, classes="data-table", escape=True, justify="left")


def svg_line_chart(
    points: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str,
    width: int = 640,
    height: int = 360,
) -> str:
    plot_left, plot_top, plot_right, plot_bottom = 58, 34, width - 22, height - 46
    frame = points.dropna(subset=[x_col, y_col]).copy()
    if frame.empty:
        return f"<div class='empty-chart'>{html.escape(title)}: no data</div>"

    xs = frame[x_col].astype(float).to_numpy()
    ys = frame[y_col].astype(float).to_numpy()
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0

    def sx(x: float) -> float:
        return plot_left + (x - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(y: float) -> float:
        return plot_bottom - (y - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    circles = "\n".join(
        f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4'><title>p={x:.3f}, freq={y:.3f}</title></circle>"
        for x, y in zip(xs, ys)
    )
    diag = f"{sx(0):.1f},{sy(0):.1f} {sx(1):.1f},{sy(1):.1f}"
    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
      <text x="{plot_left}" y="22" class="chart-title">{html.escape(title)}</text>
      <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis" />
      <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis" />
      <polyline points="{diag}" class="diag" />
      <polyline points="{polyline}" class="series" />
      {circles}
      <text x="{plot_left}" y="{height - 12}" class="axis-label">Mean Kalshi probability</text>
      <text x="12" y="{plot_top + 12}" class="axis-label">Observed</text>
      <text x="{plot_left - 24}" y="{plot_bottom + 4}" class="tick">0</text>
      <text x="{plot_right - 8}" y="{plot_bottom + 18}" class="tick">1</text>
      <text x="{plot_left - 32}" y="{plot_top + 4}" class="tick">1</text>
    </svg>
    """


def svg_histogram(
    df: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    bins: int = 20,
    width: int = 640,
    height: int = 360,
) -> str:
    values = df[value_col].dropna().astype(float).to_numpy()
    if values.size == 0:
        return f"<div class='empty-chart'>{html.escape(title)}: no data</div>"

    counts, edges = np.histogram(values, bins=np.linspace(0, 1, bins + 1))
    max_count = max(int(counts.max()), 1)
    plot_left, plot_top, plot_right, plot_bottom = 58, 34, width - 22, height - 46
    bar_gap = 2
    bar_width = (plot_right - plot_left) / bins - bar_gap

    bars = []
    for i, count in enumerate(counts):
        x = plot_left + i * ((plot_right - plot_left) / bins)
        h = (count / max_count) * (plot_bottom - plot_top)
        y = plot_bottom - h
        bars.append(
            f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_width:.1f}' height='{h:.1f}'>"
            f"<title>{edges[i]:.2f}-{edges[i + 1]:.2f}: {int(count)}</title></rect>"
        )

    return f"""
    <svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
      <text x="{plot_left}" y="22" class="chart-title">{html.escape(title)}</text>
      <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis" />
      <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis" />
      {''.join(bars)}
      <text x="{plot_left}" y="{height - 12}" class="axis-label">Kalshi probability</text>
      <text x="12" y="{plot_top + 12}" class="axis-label">Count</text>
      <text x="{plot_left - 24}" y="{plot_bottom + 4}" class="tick">0</text>
      <text x="{plot_right - 8}" y="{plot_bottom + 18}" class="tick">1</text>
      <text x="{plot_left - 42}" y="{plot_top + 4}" class="tick">{max_count:,}</text>
    </svg>
    """


def plotly_report_charts(raw: pd.DataFrame, calibration: pd.DataFrame) -> str:
    overall_cal = calibration[(calibration["segment"] == "overall") & (calibration["n_forecasts"] > 0)]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calibration", "Kalshi Probability Distribution"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#9ca3af", dash="dash"),
            name="Perfect calibration",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=overall_cal["mean_p_kalshi"],
            y=overall_cal["observed_frequency"],
            mode="lines+markers",
            name="Observed",
            line=dict(color="#2563eb"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Histogram(x=raw["p_kalshi"], nbinsx=20, marker_color="#0f766e", name="Forecasts"),
        row=1,
        col=2,
    )
    fig.update_xaxes(range=[0, 1], title_text="Mean Kalshi probability", row=1, col=1)
    fig.update_yaxes(range=[0, 1], title_text="Observed frequency", row=1, col=1)
    fig.update_xaxes(range=[0, 1], title_text="Kalshi probability", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_layout(template="plotly_white", height=420, showlegend=False, margin=dict(l=40, r=30, t=70, b=40))
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def fallback_report_charts(raw: pd.DataFrame, calibration: pd.DataFrame) -> str:
    overall_cal = calibration[(calibration["segment"] == "overall") & (calibration["n_forecasts"] > 0)]
    return (
        "<div class='chart-grid'>"
        + svg_line_chart(
            overall_cal,
            x_col="mean_p_kalshi",
            y_col="observed_frequency",
            title="Calibration",
        )
        + svg_histogram(raw, value_col="p_kalshi", title="Kalshi Probability Distribution")
        + "</div>"
    )


def build_summary_html(
    *,
    raw: pd.DataFrame,
    metrics: pd.DataFrame,
    decomposition: pd.DataFrame,
    calibration: pd.DataFrame,
    expanded_summary: pd.DataFrame,
    sharpness: pd.DataFrame,
    coverage: pd.DataFrame,
    resolution_mismatches: pd.DataFrame,
    messages: Iterable[str],
    include_coverage: bool = False,
) -> str:
    overall = metrics.loc[metrics["segment"] == "overall"].iloc[0]
    ece = expanded_summary.loc[expanded_summary["segment"] == "overall"].iloc[0]
    sharp = sharpness.loc[sharpness["segment"] == "overall"].iloc[0]

    cards = [
        ("Scored rows", fmt_num(overall["n_forecasts"], 0)),
        ("Event contracts", fmt_num(overall["n_event_contracts"], 0)),
        ("Brier score", fmt_num(overall["brier_score"])),
        ("Log loss", fmt_num(overall["log_loss"])),
        ("ECE", fmt_num(ece["expected_calibration_error"])),
        ("Audit mismatches", fmt_num(len(resolution_mismatches), 0)),
    ]
    card_html = "\n".join(
        f"<section class='metric-card'><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></section>"
        for label, value in cards
    )
    chart_html = plotly_report_charts(raw, calibration) if HAS_PLOTLY else fallback_report_charts(raw, calibration)
    message_html = "".join(f"<li>{html.escape(str(message))}</li>" for message in messages)
    assumption_html = "".join(f"<li>{html.escape(assumption)}</li>" for assumption in ASSUMPTIONS)
    plot_note = (
        "Plotly was available, so charts are interactive."
        if HAS_PLOTLY
        else "Plotly was not installed in the current environment, so this report used built-in SVG charts."
    )
    coverage_section = (
        f"""
  <h2>Outcome Join Coverage</h2>
  <div class="table-wrap">{dataframe_to_html_table(coverage)}</div>
"""
        if include_coverage
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model K Summary</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #657085;
      --line: #d7dde8;
      --soft: #f5f7fb;
      --blue: #2563eb;
      --teal: #0f766e;
      --amber: #b45309;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .metric-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--soft);
    }}
    .metric-card span {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .metric-card strong {{
      display: block;
      font-size: 24px;
      letter-spacing: 0;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .chart {{
      width: 100%;
      min-height: 280px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    .chart-title {{
      font-size: 16px;
      font-weight: 650;
      fill: var(--ink);
    }}
    .axis {{
      stroke: #94a3b8;
      stroke-width: 1;
    }}
    .diag {{
      fill: none;
      stroke: #9ca3af;
      stroke-width: 1.5;
      stroke-dasharray: 5 5;
    }}
    .series {{
      fill: none;
      stroke: var(--blue);
      stroke-width: 3;
    }}
    circle {{
      fill: var(--blue);
    }}
    rect {{
      fill: var(--teal);
    }}
    .axis-label, .tick {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
    }}
    .data-table th {{
      background: var(--soft);
      color: #334155;
      font-weight: 650;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 2px 5px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Model K: Kalshi Probability Evaluation</h1>
  <p>
    Forecasts are minute-level Kalshi YES prices from the hourly price files. Reality is the
    official Kalshi contract result for the exact matched market, encoded as <code>1</code> for
    YES resolved and <code>0</code> for NO resolved.
  </p>
  <h2>Assumptions</h2>
  <ul>{assumption_html}</ul>
  <div class="cards">{card_html}</div>
  {chart_html}
  <p class="note">{html.escape(plot_note)}</p>

  <h2>Metric Summary</h2>
  <div class="table-wrap">{dataframe_to_html_table(metrics)}</div>
  {coverage_section}

  <h2>Brier Decomposition</h2>
  <div class="table-wrap">{dataframe_to_html_table(decomposition)}</div>

  <h2>Expanded Calibration Error</h2>
  <div class="table-wrap">{dataframe_to_html_table(expanded_summary)}</div>

  <h2>Sharpness</h2>
  <div class="table-wrap">{dataframe_to_html_table(sharpness)}</div>

  <h2>Resolution Audit Mismatches</h2>
  <p class="note">
    Binance audit checks are diagnostic only and are not used as the scored truth label.
  </p>
  <div class="table-wrap">{dataframe_to_html_table(resolution_mismatches, max_rows=25)}</div>

  <h2>Output Files</h2>
  <p class="note">
    Individual CSV outputs are saved next to this report in <code>Model_K_outputs</code>:
    raw values, Brier score, Brier decomposition, log loss, calibration curve,
    expanded calibration error, sharpness, outcome join coverage, unmatched rows, and audit mismatches.
  </p>
  <ul>{message_html}</ul>
</main>
</body>
</html>
"""


def time_bucket_calibration_charts(calibration: pd.DataFrame) -> str:
    nonempty = calibration[calibration["n_forecasts"] > 0].copy()
    if nonempty.empty:
        return "<div class='empty-chart'>No time bucket calibration data</div>"

    charts = []
    for _, bucket_frame in nonempty.groupby("display_name", sort=False):
        title = str(bucket_frame["display_name"].iloc[0])
        charts.append(
            svg_line_chart(
                bucket_frame,
                x_col="mean_p_kalshi",
                y_col="observed_frequency",
                title=title,
                width=520,
                height=300,
            )
        )
    return "<div class='chart-grid'>" + "".join(charts) + "</div>"


def svg_metric_by_time_bucket(
    metrics: pd.DataFrame,
    *,
    value_col: str,
    title: str,
    y_label: str,
    width: int = 860,
    height: int = 360,
) -> str:
    frame = metrics[(metrics["bucket_type"] == "decile") & (metrics["n_forecasts"] > 0)].copy()
    frame = frame.sort_values("lower_minute")
    if frame.empty:
        return f"<div class='empty-chart'>No 10-minute {html.escape(y_label)} data</div>"

    plot_left, plot_top, plot_right, plot_bottom = 64, 34, width - 28, height - 70
    labels = frame["display_name"].astype(str).tolist()
    values = frame[value_col].astype(float).to_numpy()
    max_y = max(float(values.max()) * 1.15, 0.01)
    step = (plot_right - plot_left) / max(len(values) - 1, 1)

    def sx(i: int) -> float:
        return plot_left + i * step

    def sy(value: float) -> float:
        return plot_bottom - (value / max_y) * (plot_bottom - plot_top)

    points = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(values))
    circles = "\n".join(
        (
            f"<circle cx='{sx(i):.1f}' cy='{sy(v):.1f}' r='4'>"
            f"<title>{html.escape(labels[i])}: {v:.5f}</title></circle>"
        )
        for i, v in enumerate(values)
    )
    x_labels = "\n".join(
        f"<text x='{sx(i):.1f}' y='{plot_bottom + 22}' class='tick' text-anchor='middle'>{html.escape(label.replace('Minutes ', ''))}</text>"
        for i, label in enumerate(labels)
    )
    y_ticks = []
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        value = max_y * frac
        y = sy(value)
        y_ticks.append(
            f"<line x1='{plot_left}' y1='{y:.1f}' x2='{plot_right}' y2='{y:.1f}' class='grid' />"
            f"<text x='{plot_left - 10}' y='{y + 4:.1f}' class='tick' text-anchor='end'>{value:.3f}</text>"
        )

    return f"""
    <svg class="chart wide-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
      <text x="{plot_left}" y="22" class="chart-title">{html.escape(title)}</text>
      {''.join(y_ticks)}
      <line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis" />
      <line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis" />
      <polyline points="{points}" class="series" />
      {circles}
      {x_labels}
      <text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 16}" class="axis-label" text-anchor="middle">Market Minute</text>
      <text x="16" y="{plot_top + 12}" class="axis-label">{html.escape(y_label)}</text>
    </svg>
    """


def svg_brier_by_time_bucket(metrics: pd.DataFrame, *, width: int = 860, height: int = 360) -> str:
    return svg_metric_by_time_bucket(
        metrics,
        value_col="brier_score",
        title="Brier Score By 10-Minute Bucket",
        y_label="Brier",
        width=width,
        height=height,
    )


def svg_brier_decomposition_by_time_bucket(brier: pd.DataFrame) -> str:
    reliability = svg_metric_by_time_bucket(
        brier,
        value_col="reliability",
        title="Reliability By 10-Minute Bucket",
        y_label="Reliability",
        width=520,
        height=300,
    )
    resolution = svg_metric_by_time_bucket(
        brier,
        value_col="resolution",
        title="Resolution By 10-Minute Bucket",
        y_label="Resolution",
        width=520,
        height=300,
    )
    return f"<div class='chart-grid compact-chart-grid'>{reliability}{resolution}</div>"


def build_time_bucket_summary_html(
    *,
    time_bucket_metrics: pd.DataFrame,
    time_bucket_accuracy: pd.DataFrame,
    time_bucket_brier: pd.DataFrame,
    time_bucket_calibration: pd.DataFrame,
) -> str:
    nonempty_metrics = time_bucket_metrics[time_bucket_metrics["n_forecasts"] > 0].copy()
    isolated_metrics = nonempty_metrics[nonempty_metrics["bucket_type"] == "isolated"].copy()
    decile_metrics = nonempty_metrics[nonempty_metrics["bucket_type"] == "decile"].copy()
    isolated_accuracy = time_bucket_accuracy[
        (time_bucket_accuracy["bucket_type"] == "isolated") & (time_bucket_accuracy["n_forecasts"] > 0)
    ].copy()
    decile_accuracy = time_bucket_accuracy[
        (time_bucket_accuracy["bucket_type"] == "decile") & (time_bucket_accuracy["n_forecasts"] > 0)
    ].copy()

    metrics_cols = [
        "display_name",
        "n_forecasts",
        "n_event_contracts",
        "base_rate",
        "mean_p_kalshi",
        "brier_score",
        "log_loss",
        "classification_accuracy",
    ]
    accuracy_cols = [
        "display_name",
        "n_forecasts",
        "correct_forecasts",
        "incorrect_forecasts",
        "classification_accuracy_pct",
        "threshold",
    ]
    brier_cols = [
        "display_name",
        "n_forecasts",
        "brier_score",
        "reliability",
        "resolution",
        "uncertainty",
        "brier_from_decomposition",
    ]

    brier_chart_html = svg_brier_by_time_bucket(time_bucket_metrics)
    brier_decomposition_chart_html = svg_brier_decomposition_by_time_bucket(time_bucket_brier)
    chart_html = time_bucket_calibration_charts(time_bucket_calibration)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model K Market Minute Bucket Summary</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #172033;
      --muted: #657085;
      --line: #d7dde8;
      --soft: #f5f7fb;
      --blue: #2563eb;
      --teal: #0f766e;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: white;
    }}
    main {{
      max-width: 1240px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 34px 0 12px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    p, li {{
      color: var(--muted);
      line-height: 1.45;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 18px;
      align-items: start;
    }}
    .compact-chart-grid {{
      margin-bottom: 18px;
    }}
    .chart {{
      width: 100%;
      min-height: 240px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
    }}
    .wide-chart {{
      min-height: 320px;
      margin-bottom: 18px;
    }}
    .chart-title {{
      font-size: 15px;
      font-weight: 650;
      fill: var(--ink);
    }}
    .axis {{
      stroke: #94a3b8;
      stroke-width: 1;
    }}
    .grid {{
      stroke: #e2e8f0;
      stroke-width: 1;
    }}
    .diag {{
      fill: none;
      stroke: #9ca3af;
      stroke-width: 1.5;
      stroke-dasharray: 5 5;
    }}
    .series {{
      fill: none;
      stroke: var(--blue);
      stroke-width: 3;
    }}
    circle {{
      fill: var(--blue);
    }}
    .axis-label, .tick {{
      fill: var(--muted);
      font-size: 12px;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 18px;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      white-space: nowrap;
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
    }}
    .data-table th {{
      background: var(--soft);
      color: #334155;
      font-weight: 650;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
    }}
    code {{
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 5px;
      padding: 2px 5px;
    }}
  </style>
</head>
<body>
<main>
  <h1>Model K: Market Minute Buckets</h1>
  <p>
    Buckets are based on elapsed market minute, where source <code>minute_number = 0</code> is
    displayed as <code>Minute 1</code>. Minute 1 and minute 30 are isolated single-minute views;
    the 10-minute buckets are inclusive elapsed-market ranges such as 1-10, 11-20, and so on.
    These bucket views intentionally overlap for isolated minute 1 and minute 30.
  </p>

  <h2>Isolated Minute Metrics</h2>
  <div class="table-wrap">{dataframe_to_html_table(isolated_metrics[metrics_cols])}</div>

  <h2>10-Minute Bucket Metrics</h2>
  {brier_chart_html}
  {brier_decomposition_chart_html}
  <div class="table-wrap">{dataframe_to_html_table(decile_metrics[metrics_cols])}</div>

  <h2>Accuracy Evaluation</h2>
  <p class="note">Accuracy uses the same threshold classification rule as the main summary.</p>
  <div class="table-wrap">{dataframe_to_html_table(pd.concat([isolated_accuracy, decile_accuracy], ignore_index=True)[accuracy_cols])}</div>

  <h2>Brier Decomposition By Bucket</h2>
  <div class="table-wrap">{dataframe_to_html_table(time_bucket_brier[time_bucket_brier["n_forecasts"] > 0][brier_cols])}</div>

  <h2>Calibration Curves</h2>
  {chart_html}

  <h2>Output Files</h2>
  <p class="note">
    Detailed bucket data is saved in <code>time_bucket_metrics.csv</code>,
    <code>time_bucket_accuracy.csv</code>, <code>time_bucket_brier_decomposition.csv</code>, and
    <code>time_bucket_calibration_curve.csv</code>.
  </p>
</main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    forecasts = load_kalshi_price_outputs(args.kalshi_price_dir.resolve())
    settlements = load_settlements(args.settlement_csv.resolve())
    outcomes, messages = build_kalshi_reality_outcomes(
        forecasts=forecasts,
        settlements=settlements,
        output_dir=output_dir,
    )
    raw, unmatched = attach_outcomes(forecasts, outcomes)
    coverage = build_outcome_join_coverage(forecasts, raw, unmatched)
    resolution_mismatches = pd.DataFrame() if args.skip_binance_audit else build_resolution_mismatches(raw)

    overall_coverage = coverage.loc[coverage["scope"] == "overall"].iloc[0]
    messages.append(
        f"Matched {int(overall_coverage['matched_rows'])} of "
        f"{int(overall_coverage['total_forecast_rows'])} forecast rows to official Kalshi outcomes."
    )
    if len(resolution_mismatches):
        messages.append(
            f"Found {len(resolution_mismatches)} Binance audit mismatch row(s); see resolution_mismatches.csv."
        )

    metrics = build_metrics_summary(raw, threshold=args.classification_threshold)
    decomposition = build_brier_decomposition(raw, bins=args.calibration_bins)
    calibration, expanded_summary = expanded_calibration_error(raw, bins=args.calibration_bins)
    sharpness = build_sharpness(raw)
    time_bucket_metrics, time_bucket_accuracy, time_bucket_brier, time_bucket_calibration = build_time_bucket_outputs(
        raw,
        calibration_bins=args.calibration_bins,
        threshold=args.classification_threshold,
    )

    write_individual_metric_files(
        output_dir=output_dir,
        raw=raw,
        metrics=metrics,
        decomposition=decomposition,
        calibration=calibration,
        expanded_error=calibration,
        expanded_error_summary=expanded_summary,
        sharpness=sharpness,
        coverage=coverage,
        unmatched=unmatched,
        resolution_mismatches=resolution_mismatches,
        time_bucket_metrics=time_bucket_metrics,
        time_bucket_accuracy=time_bucket_accuracy,
        time_bucket_brier=time_bucket_brier,
        time_bucket_calibration=time_bucket_calibration,
    )

    summary_html = build_summary_html(
        raw=raw,
        metrics=metrics,
        decomposition=decomposition,
        calibration=calibration,
        expanded_summary=expanded_summary,
        sharpness=sharpness,
        coverage=coverage,
        resolution_mismatches=resolution_mismatches,
        messages=messages,
        include_coverage=False,
    )
    summary_path = output_dir / "model_k_summary.html"
    summary_path.write_text(summary_html, encoding="utf-8")

    summary_with_coverage_html = build_summary_html(
        raw=raw,
        metrics=metrics,
        decomposition=decomposition,
        calibration=calibration,
        expanded_summary=expanded_summary,
        sharpness=sharpness,
        coverage=coverage,
        resolution_mismatches=resolution_mismatches,
        messages=messages,
        include_coverage=True,
    )
    summary_with_coverage_path = output_dir / "model_k_summary_with_coverage.html"
    summary_with_coverage_path.write_text(summary_with_coverage_html, encoding="utf-8")

    time_bucket_summary_html = build_time_bucket_summary_html(
        time_bucket_metrics=time_bucket_metrics,
        time_bucket_accuracy=time_bucket_accuracy,
        time_bucket_brier=time_bucket_brier,
        time_bucket_calibration=time_bucket_calibration,
    )
    time_bucket_summary_path = output_dir / "time_bucket_summary.html"
    time_bucket_summary_path.write_text(time_bucket_summary_html, encoding="utf-8")

    print(f"Model K outputs saved to: {output_dir}")
    print(f"Summary report: {summary_path}")
    print(f"Summary report with coverage: {summary_with_coverage_path}")
    print(f"Time bucket report: {time_bucket_summary_path}")
    for message in messages:
        print(message)


if __name__ == "__main__":
    main()
