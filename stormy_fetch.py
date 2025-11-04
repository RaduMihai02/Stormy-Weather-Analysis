import sys
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import requests
import unicodedata
import os

# - Guide for readers:
# - This file fetches weather for 12 Romanian cities, cleans it, and saves CSVs.
# - We combine the last 14 days (history) with the next 14 days (forecast).
# - Then we compute helpful fields (e.g., hourly "state": sunny/cloudy/rain/snow/thunder),
#   roll up to daily summaries, and also produce longer-term historical summaries.

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    # Python <3.9 fallback if needed
    from backports.zoneinfo import ZoneInfo  # type: ignore


# ----------------------
# Configuration
# ----------------------

BUCHAREST_TZ = ZoneInfo("Europe/Bucharest")
# How many days back we look from today for recent history in the main view
HIST_DAYS = 14
# How many days forward we request from the forecast API (we later trim to >= today)
FORECAST_DAYS = 14

# Where to write CSV outputs (created automatically if missing)
OUT_DIR = "CSVs"

# Thresholds (can be tweaked later)
CLOUDY_THRESHOLD = 60  # % cloud cover (60+% looks cloudy)
SUNNY_THRESHOLD = 25   # % cloud cover (25-% looks sunny)
RAINY_HOURLY_PROB_THRESHOLD_MM = 0.1  # If historical hourly precip > 0.1 mm, we treat that hour as rainy (100% chance)
RAINY_DAY_THRESHOLD_MM = 0.0  # Any measurable rain (>0.0 mm) counts the day as a rainy day in historical summaries


# Our 12 major Romanian cities (display names as you want to see them in UI and CSV)
# The 12 cities we include in the app (display names used in CSV/UI)
CITIES = [
    "București",
    "Cluj-Napoca",
    "Timișoara",
    "Iași",
    "Constanța",
    "Craiova",
    "Brașov",
    "Galați",
    "Ploiești",
    "Oradea",
    "Sibiu",
    "Bacău",
]


@dataclass
class CityCoord:
    # A simple container that holds the name and GPS coordinates of a city
    name: str
    latitude: float
    longitude: float


def bucharest_today_start() -> datetime:
    # We anchor “today” to midnight in Bucharest time so history/forecast split is consistent
    now = datetime.now(BUCHAREST_TZ)
    return datetime(year=now.year, month=now.month, day=now.day, tzinfo=BUCHAREST_TZ)


def get_city_coordinates(city: str, session: Optional[requests.Session] = None) -> CityCoord:
    """Resolve the city's latitude/longitude using Open‑Meteo's geocoding service.

    We also try an ASCII fallback (remove diacritics) if the first attempt returns nothing.
    Restricting to country RO avoids ambiguous matches outside Romania.
    """
    s = session or requests.Session()
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1,
        "language": "en",
        "format": "json",
        "country": "RO",
    }
    data = _request_json(s, url, params=params, timeout=30)
    results = data.get("results") or []
    if not results:
        # Try ASCII fallback (strip diacritics)
        city_ascii = _asciiize(city)
        if city_ascii != city:
            params_ascii = dict(params)
            params_ascii["name"] = city_ascii
            data2 = _request_json(s, url, params=params_ascii, timeout=30)
            results = data2.get("results") or []
    if not results:
        raise RuntimeError(f"No geocoding result for city: {city}")
    first = results[0]
    return CityCoord(name=city, latitude=float(first["latitude"]), longitude=float(first["longitude"]))


def fetch_archive_hourly(city: CityCoord, start_date: date, end_date: date, session: Optional[requests.Session] = None) -> pd.DataFrame:
    # Download past hourly weather between start_date and end_date (inclusive)
    # We request only the variables we later need for our plots and daily rollups.
    url = "https://archive-api.open-meteo.com/v1/archive"
    hourly_vars = [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "cloudcover",
        "windspeed_10m",
        "windgusts_10m",
        "relativehumidity_2m",
        "weathercode",
    ]
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(hourly_vars),
        "timezone": "Europe/Bucharest",
        "windspeed_unit": "kmh",
    }
    s = session or requests.Session()
    js = _request_json(s, url, params=params, timeout=60)
    hourly = js.get("hourly") or {}
    if not hourly:
        return pd.DataFrame(columns=["datetime_local"] + hourly_vars)
    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "datetime_local"}, inplace=True)
    df["city"] = city.name
    # Ensure expected columns exist (sometimes APIs omit a column when all values are missing)
    for col in hourly_vars:
        if col not in df.columns:
            df[col] = np.nan
    # cast types if possible
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
    return df[["city", "datetime_local"] + hourly_vars]


def fetch_forecast_hourly(city: CityCoord, days: int, session: Optional[requests.Session] = None) -> pd.DataFrame:
    # Download upcoming hourly forecast for the requested number of days.
    # This includes precipitation probability which the archive API does not provide.
    url = "https://api.open-meteo.com/v1/forecast"
    hourly_vars = [
        "temperature_2m",
        "precipitation",
        "rain",
        "snowfall",
        "cloudcover",
        "windspeed_10m",
        "windgusts_10m",
        "relativehumidity_2m",
        "weathercode",
        "precipitation_probability",
    ]
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "hourly": ",".join(hourly_vars),
        "forecast_days": days,
        "timezone": "Europe/Bucharest",
        "windspeed_unit": "kmh",
    }
    s = session or requests.Session()
    js = _request_json(s, url, params=params, timeout=60)
    hourly = js.get("hourly") or {}
    if not hourly:
        return pd.DataFrame(columns=["datetime_local"] + hourly_vars)
    df = pd.DataFrame(hourly)
    df.rename(columns={"time": "datetime_local"}, inplace=True)
    df["city"] = city.name
    for col in hourly_vars:
        if col not in df.columns:
            df[col] = np.nan
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
    return df[["city", "datetime_local"] + hourly_vars]


def derive_hourly_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw API columns into consistent, human‑friendly hourly fields.

    - Rename columns to stable names (e.g., temperature_2m → temp_c)
    - Fill historical rain probability (archive lacks it) with a simple rule
    - Compute an easy to read “weather_state” per hour (sunny/cloudy/rain/etc.)
    - Return only the columns we plot/use, sorted by city/time
    """
    df = df.copy()

    # Standardize column names for outputs
    df.rename(
        columns={
            "temperature_2m": "temp_c",
            "precipitation": "precip_mm",
            "rain": "rain_mm",
            "snowfall": "snowfall_mm",
            "cloudcover": "cloudcover_pct",
            "windspeed_10m": "wind_kmh",
            "windgusts_10m": "wind_gust_kmh",
            "relativehumidity_2m": "humidity_pct",
            "precipitation_probability": "precip_prob_pct",
        },
        inplace=True,
    )

    # Historical hours do not include rain probability from the API.
    # We fill it: if that hour had > 0.1 mm precipitation, we mark 100% rain chance; else 0%.
    hist_mask = df["precip_prob_pct"].isna()
    if hist_mask.any():
        df.loc[hist_mask, "precip_prob_pct"] = np.where(
            (df.loc[hist_mask, "precip_mm"].astype(float) > RAINY_HOURLY_PROB_THRESHOLD_MM), 100.0, 0.0
        )

    # Determine a simplified weather state per hour (used for visuals and daily vote)
    df["weather_state"] = df.apply(_determine_state_row, axis=1)

    # Keep and order output columns
    out_cols = [
        "city",
        "datetime_local",
        "temp_c",
        "precip_prob_pct",
        "precip_mm",
        "rain_mm",
        "snowfall_mm",
        "cloudcover_pct",
        "wind_kmh",
        "wind_gust_kmh",
        "humidity_pct",
        "weather_state",
    ]
    return df[out_cols].sort_values(["city", "datetime_local"]).reset_index(drop=True)


def _determine_state_row(row: pd.Series) -> str:
    """Map an hour to one of: sunny, cloudy, rain, thunderstorm, snowstorm.

    We use a clear priority so only one label wins:
    thunderstorm → snowstorm → rain → cloudy → sunny
    """
    code = _safe_int(row.get("weathercode"))
    cloud = _safe_float(row.get("cloudcover_pct"))
    precip = _safe_float(row.get("precip_mm"))
    rain = _safe_float(row.get("rain_mm"))
    snow = _safe_float(row.get("snowfall_mm"))

    # Thunderstorm: specific weather codes (95–99) override everything
    if code is not None and 95 <= code <= 99:
        return "thunderstorm"

    # Snow: any snowfall amount or a snow-related code
    if (snow is not None and snow > 0) or (code in {71, 72, 73, 75, 77, 85, 86}):
        return "snowstorm"

    # Rain: if either rain_mm or precip_mm is positive, or code indicates rain
    if (rain is not None and rain > 0) or (precip is not None and precip > 0) or (
        code in {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82}
    ):
        return "rain"

    # Otherwise decide by cloud cover thresholds
    if cloud is not None:
        if cloud >= CLOUDY_THRESHOLD:
            return "cloudy"
        if cloud <= SUNNY_THRESHOLD:
            return "sunny"

    # When missing data or in-between values, fall back to "cloudy"
    return "cloudy"


def _safe_float(x) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return int(x)
    except Exception:
        return None


def build_daily_from_hourly(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Roll up hours into daily summaries that are easy to compare between cities.

    - Compute daily mean/min/max temperature
    - Sum daily precipitation and average the daily rain probability
    - Pick a daily “weather_state_day” via majority of hourly states (with a clear tie-breaker)
    """
    if df_hourly.empty:
        return pd.DataFrame(
            columns=[
                "city",
                "date_local",
                "temp_mean_c",
                "temp_min_c",
                "temp_max_c",
                "precip_sum_mm",
                "precip_prob_avg_pct",
                "weather_state_day",
            ]
        )

    df = df_hourly.copy()
    df["date_local"] = df["datetime_local"].dt.date

    agg_temp = df.groupby(["city", "date_local"])['temp_c'].agg(['mean', 'min', 'max']).rename(
        columns={"mean": "temp_mean_c", "min": "temp_min_c", "max": "temp_max_c"}
    )
    precip_sum = df.groupby(["city", "date_local"])['precip_mm'].sum().rename("precip_sum_mm")
    prob_avg = df.groupby(["city", "date_local"])['precip_prob_pct'].mean().rename("precip_prob_avg_pct")

    # Majority vote for daily weather state with a deterministic tie-break
    state_counts = df.groupby(["city", "date_local", "weather_state"]).size().rename("count").reset_index()
    # pick state with highest count; break ties via priority order
    priority = {"thunderstorm": 5, "snowstorm": 4, "rain": 3, "cloudy": 2, "sunny": 1}
    state_counts["priority"] = state_counts["weather_state"].map(priority)
    state_counts.sort_values(["city", "date_local", "count", "priority"], ascending=[True, True, False, False], inplace=True)
    top_state = state_counts.groupby(["city", "date_local"]).first().reset_index()[["city", "date_local", "weather_state"]]
    top_state.rename(columns={"weather_state": "weather_state_day"}, inplace=True)

    daily = (
        agg_temp
        .join(precip_sum)
        .join(prob_avg)
        .reset_index()
        .merge(top_state, on=["city", "date_local"], how="left")
    )

    # Arrange columns and sort for predictable output
    daily = daily[[
        "city",
        "date_local",
        "temp_mean_c",
        "temp_min_c",
        "temp_max_c",
        "precip_sum_mm",
        "precip_prob_avg_pct",
        "weather_state_day",
    ]]

    # Sort
    daily.sort_values(["city", "date_local"], inplace=True)
    daily.reset_index(drop=True, inplace=True)
    return daily


# ----------------------
# Historical (daily) fetching & aggregation
# These functions power the monthly/quarterly context views and the projection.
# ----------------------

def fetch_archive_daily(city: CityCoord, start_date: date, end_date: date, session: Optional[requests.Session] = None) -> pd.DataFrame:
    """Fetch daily values for a date window from the Archive API.

    We request only what we need to compute clear long‑term comparisons (min/max temp, precipitation sum).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    daily_vars = [
        "temperature_2m_min",
        "temperature_2m_max",
        "precipitation_sum",
    ]
    params = {
        "latitude": city.latitude,
        "longitude": city.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(daily_vars),
        "timezone": "Europe/Bucharest",
    }
    s = session or requests.Session()
    js = _request_json(s, url, params=params, timeout=60)
    daily = js.get("daily") or {}
    if not daily:
        return pd.DataFrame(columns=["date_local"] + daily_vars)
    df = pd.DataFrame(daily)
    df.rename(columns={"time": "date_local"}, inplace=True)
    df["city"] = city.name
    # Ensure expected columns
    for col in daily_vars:
        if col not in df.columns:
            df[col] = np.nan
    # Cast types
    df["date_local"] = pd.to_datetime(df["date_local"]).dt.date
    return df[["city", "date_local"] + daily_vars]


def build_historical_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Produce historical daily metrics used in monthly/quarterly summaries.

    - temp_mean_c is the midpoint between min and max
    - rained_day is True if precipitation_sum > 0.0 mm (any measurable rain)
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "city", "date_local", "temp_min_c", "temp_max_c", "temp_mean_c", "rained_day"
        ])
    out = df.copy()
    out.rename(columns={
        "temperature_2m_min": "temp_min_c",
        "temperature_2m_max": "temp_max_c",
        "precipitation_sum": "precip_sum_mm",
    }, inplace=True)
    out["temp_mean_c"] = (out["temp_min_c"].astype(float) + out["temp_max_c"].astype(float)) / 2.0
    # Rainy day = any precipitation > threshold
    out["rained_day"] = (out["precip_sum_mm"].astype(float) > RAINY_DAY_THRESHOLD_MM)
    out = out[["city", "date_local", "temp_min_c", "temp_max_c", "temp_mean_c", "rained_day"]]
    out.sort_values(["city", "date_local"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _asciiize(text: str) -> str:
    """Remove Romanian diacritics/accents so we can try an ASCII geocoding fallback when needed."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _request_json(session: requests.Session, url: str, params: Optional[dict] = None, timeout: int = 60, retries: int = 3, backoff: float = 0.5) -> dict:
    """Download a JSON response with a few safe retries.

    This makes the app resilient to temporary network hiccups and services that respond slowly.
    """
    last_exc: Optional[Exception] = None
    headers = {"Accept": "application/json"}
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            # Even if content-type is not JSON, try to parse; may raise
            return resp.json()
        except Exception as e:
            last_exc = e
            # small backoff
            time.sleep(backoff * (attempt + 1))
            continue
    # If we captured an HTML/text error, surface a short snippet for debugging
    try:
        # Best-effort second request to capture body without retry loop
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        snippet = resp.text[:200]
        raise RuntimeError(f"Failed JSON fetch after {retries} retries at {url}. Last error: {last_exc}. Body starts with: {snippet}")
    except Exception:
        # Fallback to last exception only
        if last_exc:
            raise last_exc
        raise


def aggregate_monthly(df_daily_hist: pd.DataFrame) -> pd.DataFrame:
    # Group historical daily data by year/month and compute average/limits + rainy-day fraction
    if df_daily_hist.empty:
        return pd.DataFrame(columns=[
            "city", "year", "month", "temp_mean_c", "temp_min_c", "temp_max_c", "rainy_days", "rainy_day_frac"
        ])
    df = df_daily_hist.copy()
    df["date_local"] = pd.to_datetime(df["date_local"])
    df["year"] = df["date_local"].dt.year
    df["month"] = df["date_local"].dt.month

    grouped = df.groupby(["city", "year", "month"])
    temp_mean = grouped["temp_mean_c"].mean().rename("temp_mean_c")
    temp_min = grouped["temp_min_c"].min().rename("temp_min_c")
    temp_max = grouped["temp_max_c"].max().rename("temp_max_c")
    rainy_days = grouped["rained_day"].sum().rename("rainy_days")
    days_count = grouped["rained_day"].count().rename("days")
    rainy_frac = (rainy_days / days_count).rename("rainy_day_frac")

    monthly = pd.concat([temp_mean, temp_min, temp_max, rainy_days, rainy_frac], axis=1).reset_index()
    monthly.sort_values(["city", "year", "month"], inplace=True)
    monthly.reset_index(drop=True, inplace=True)
    return monthly


def aggregate_quarterly(df_daily_hist: pd.DataFrame) -> pd.DataFrame:
    # Group historical daily data by year/quarter and compute average/limits + rainy-day fraction
    if df_daily_hist.empty:
        return pd.DataFrame(columns=[
            "city", "year", "quarter", "temp_mean_c", "temp_min_c", "temp_max_c", "rainy_days", "rainy_day_frac"
        ])
    df = df_daily_hist.copy()
    df["date_local"] = pd.to_datetime(df["date_local"])
    # Build quarter labels Q1..Q4
    df["year"] = df["date_local"].dt.year
    df["quarter"] = "Q" + df["date_local"].dt.quarter.astype(str)

    grouped = df.groupby(["city", "year", "quarter"])
    temp_mean = grouped["temp_mean_c"].mean().rename("temp_mean_c")
    temp_min = grouped["temp_min_c"].min().rename("temp_min_c")
    temp_max = grouped["temp_max_c"].max().rename("temp_max_c")
    rainy_days = grouped["rained_day"].sum().rename("rainy_days")
    days_count = grouped["rained_day"].count().rename("days")
    rainy_frac = (rainy_days / days_count).rename("rainy_day_frac")

    quarterly = pd.concat([temp_mean, temp_min, temp_max, rainy_days, rainy_frac], axis=1).reset_index()
    quarterly.sort_values(["city", "year", "quarter"], inplace=True)
    quarterly.reset_index(drop=True, inplace=True)
    return quarterly


def fetch_city_data(city_name: str, session: Optional[requests.Session] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get everything needed for one city, ready for saving and plotting.

    Steps:
    - Resolve coordinates
    - Fetch the last 14 days (archive) and next 14 days (forecast)
    - Keep only forecast hours from today forward
    - Combine, standardize hourly fields, then build the daily rollups
    """
    coord = get_city_coordinates(city_name, session=session)

    today_start = bucharest_today_start()
    hist_start = (today_start.date() - timedelta(days=HIST_DAYS))
    hist_end = (today_start.date() - timedelta(days=1))

    df_hist = fetch_archive_hourly(coord, hist_start, hist_end, session=session)
    df_fore = fetch_forecast_hourly(coord, FORECAST_DAYS, session=session)

    # Partition forecast to >= today_start; archive already constrained to < today_start
    mask_fore = df_fore["datetime_local"] >= pd.Timestamp(today_start.replace(tzinfo=None))
    df_fore = df_fore.loc[mask_fore].copy()

    # Combine
    df_all = pd.concat([df_hist, df_fore], ignore_index=True)

    # Derive hourly fields and build daily
    df_hourly = derive_hourly_fields(df_all)
    df_daily = build_daily_from_hourly(df_hourly)
    return df_hourly, df_daily


def main():
    # This is the entry point that fetches all cities and writes the CSV files
    print("Stormy fetcher starting…")
    session = requests.Session()

    all_hourly: List[pd.DataFrame] = []
    all_daily: List[pd.DataFrame] = []

    for i, city in enumerate(CITIES, start=1):
        # Fetch both hourly and daily views for each city, with a small delay to be polite to the API
        try:
            print(f"[{i}/{len(CITIES)}] Fetching {city} …", flush=True)
            h, d = fetch_city_data(city, session=session)
            all_hourly.append(h)
            all_daily.append(d)
        except Exception as e:
            print(f"Error fetching {city}: {e}", file=sys.stderr, flush=True)
            continue
        # Small courtesy delay to be polite to API (optional)
        time.sleep(0.3)

    if not all_hourly:
        print("No data fetched. Exiting.")
        sys.exit(1)

    hourly_csv = pd.concat(all_hourly, ignore_index=True)
    daily_csv = pd.concat(all_daily, ignore_index=True)

    # Basic sanity cleanup
    hourly_csv.sort_values(["city", "datetime_local"], inplace=True)
    daily_csv.sort_values(["city", "date_local"], inplace=True)

    # Write outputs (files are overwritten each run to reflect the latest forecast/history)
    os.makedirs(OUT_DIR, exist_ok=True)
    hourly_path = os.path.join(OUT_DIR, "hourly.csv")
    daily_path = os.path.join(OUT_DIR, "daily.csv")
    hourly_csv.to_csv(hourly_path, index=False)
    daily_csv.to_csv(daily_path, index=False)

    print(f"Wrote {hourly_path} and {daily_path}")
    try:
        print(f"hourly.csv rows: {len(hourly_csv)} | daily.csv rows: {len(daily_csv)}")
        # Show a quick per-city summary
        per_city_daily = daily_csv.groupby("city")["date_local"].count().to_dict()
        print("Daily rows per city:", per_city_daily)
    except Exception:
        pass

    # ----------------------
    # Historical (daily) for 2023, 2024, and current year YTD
    # We build deeper context for the UI: daily history, then roll up to months and quarters.
    # ----------------------
    print("\nFetching historical daily data (2023, 2024, current YTD)…")
    hist_daily_frames: List[pd.DataFrame] = []
    today_start = bucharest_today_start()
    current_year = today_start.year

    for i, city in enumerate(CITIES, start=1):
        try:
            coord = get_city_coordinates(city, session=session)
            # 2023
            y2023 = fetch_archive_daily(
                coord, date(2023, 1, 1), date(2023, 12, 31), session=session
            )
            # 2024
            y2024 = fetch_archive_daily(
                coord, date(2024, 1, 1), date(2024, 12, 31), session=session
            )
            # Current year YTD (Jan 1 -> today)
            ytd = fetch_archive_daily(
                coord, date(current_year, 1, 1), today_start.date(), session=session
            )
            df_city = pd.concat([y2023, y2024, ytd], ignore_index=True)
            df_city = build_historical_daily(df_city)
            hist_daily_frames.append(df_city)
            print(f"[{i}/{len(CITIES)}] Historical: {city} -> {len(df_city)} rows")
        except Exception as e:
            print(f"Historical error for {city}: {e}", file=sys.stderr)
            continue
        time.sleep(0.2)

    if hist_daily_frames:
        historical_daily = pd.concat(hist_daily_frames, ignore_index=True)
        historical_daily.sort_values(["city", "date_local"], inplace=True)
        hist_daily_path = os.path.join(OUT_DIR, "historical_daily.csv")
        historical_daily.to_csv(hist_daily_path, index=False)
        print(f"Wrote {hist_daily_path} ({len(historical_daily)} rows)")

        # Aggregations
        hist_monthly = aggregate_monthly(historical_daily)
        hist_quarterly = aggregate_quarterly(historical_daily)
        hist_monthly_path = os.path.join(OUT_DIR, "historical_monthly.csv")
        hist_quarterly_path = os.path.join(OUT_DIR, "historical_quarterly.csv")
        hist_monthly.to_csv(hist_monthly_path, index=False)
        hist_quarterly.to_csv(hist_quarterly_path, index=False)
        print(
            f"Wrote {hist_monthly_path} ({len(hist_monthly)} rows) and {hist_quarterly_path} ({len(hist_quarterly)} rows)"
        )
    else:
        print("No historical data produced.")


if __name__ == "__main__":
    main()
