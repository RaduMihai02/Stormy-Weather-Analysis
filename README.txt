Stormy — Weather Insights That Answers “What Will It Feel Like?”

Why this exists (the problem)
Most weather apps flood you with raw numbers. What people need day‑to‑day is a quick sense of “what it will feel like” this week and how that compares to typical conditions. For 12 major Romanian cities, Stormy distills the last 14 days and the next 14 days into clear visuals and simple, comparable metrics — so decisions are easier and faster.

Who cares (audience)
- Commuters and travelers planning their week across cities
- Event organizers and city operations balancing comfort and risk (heat, rain, wind)
- Analysts who want reproducible, clean data that connects short‑term weather with recent history

What you get (value)
- Today at a glance: hourly temperature line, rain chance bars, and a color ribbon for the hour‑by‑hour state (sunny, cloudy, rain, thunderstorm, snowstorm).
- Next 7 days: simple min–max ranges, daily mean with uncertainty, and daily rain chance.
- Context from history: daily aggregates for 2023, 2024, and the current year‑to‑date; monthly and quarterly rollups.
- A projection (not a forecast): a baseline‑plus‑anomaly view of the remaining months so you can see how this year is tracking versus 2023/2024.

Insights this surfaces quickly
- Which city is likely to be warmest/coldest over the next week
- Where the swing between daytime highs and lows is the largest (volatility)
- Which cities have the highest near‑term rain risk
- How this year compares to 2023/2024 by quarter and month
- Whether the current year is trending warmer or wetter than the recent baseline

How it works (under the hood)
1) Geocoding: city name → coordinates using Open‑Meteo Geocoding.
2) Fetch: last 14 days (archive) + next 14 days (forecast) for hourly variables; consistent Europe/Bucharest timezone.
3) Derive: classify each hour into a simple state using weather codes, precipitation, snow, and cloud thresholds.
4) Aggregate: build daily min/max/mean, rain chance, and dominant daily state; compute historical daily for 2023, 2024, and current year‑to‑date; aggregate to monthly and quarterly.
5) Output: write CSVs for analysis and load them into a small desktop UI for exploration.

Quickstart
1) Python 3.9+
2) Install dependencies:
   pip install -r requirements.txt
3) Fetch fresh data (writes CSVs into CSVs/):
   python stormy_fetch.py
4) Open the UI (reads CSVs from CSVs/ and plots):
   python stormy_ui.py

Offline use
- If you can’t run the fetcher (no internet, restricted environment), the UI will still open using whatever CSVs are present in the CSVs/ folder.

Outputs (written/read in CSVs/)
- CSVs/hourly.csv: city, datetime_local, temp_c, precip_prob_pct, precip_mm, rain_mm, snowfall_mm, cloudcover_pct, wind_kmh, wind_gust_kmh, humidity_pct, weather_state
- CSVs/daily.csv: city, date_local, temp_mean_c, temp_min_c, temp_max_c, precip_sum_mm, precip_prob_avg_pct, weather_state_day
- CSVs/historical_daily.csv: city, date_local, temp_min_c, temp_max_c, temp_mean_c, rained_day
- CSVs/historical_monthly.csv: city, year, month, temp_mean_c, temp_min_c, temp_max_c, rainy_days, rainy_day_frac
- CSVs/historical_quarterly.csv: city, year, quarter, temp_mean_c, temp_min_c, temp_max_c, rainy_days, rainy_day_frac

Notes & assumptions
- Timezone is normalized to Europe/Bucharest in requests; timestamps are local.
- Historical rain probability is estimated as 100% if precip > 0.1 mm else 0% (archive API doesn’t provide probability).
- CSVs are overwritten on each run so the “future” always reflects the latest forecast.
- “Projection” is baseline+anomaly (not a forecast) to visualize how the current year is tracking versus 2023/2024.
- Network access is required to call Open‑Meteo APIs.

Data sources
- Open‑Meteo Geocoding, Forecast, and Archive APIs (free and no API key required).

Final takes
- The code demonstrates a reproducible data pipeline, clear transformations, and pragmatic visualization intended to answer concrete questions, not just plot lines.
- The UI choices are biased toward quick comparisons and “decision support,” not novelty.
- The project is intentionally small and readable; each step can be swapped for a different source or extended with more metrics.
