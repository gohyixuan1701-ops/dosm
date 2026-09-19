"""
cpi_score.py -- state CPI (cost of living / affordability) trend
projection, using linear regression (NOT RNN/LSTM -- see METHODOLOGY
NOTE, same reasoning as population_score.py).

METHODOLOGY NOTE (for the report):
Checked the actual monthly CPI series (2010-2026, cpi_2d_state.csv)
before choosing a method, same discipline as every other category:
- The index itself climbs smoothly and continuously (e.g. Selangor:
  99.3 in Jan 2010 to 142.1 in Jan 2026) -- no sharp reversals.
- Year-over-year inflation stayed mild throughout, including COVID
  (roughly -0.5% to +4.2%) -- nothing like GDP's -9% to +13% swing.
- There is ONE notable one-month dip (April 2020, -3.2 points,
  Selangor) -- almost certainly the COVID-era fuel/petrol price crash,
  a real but minor and temporary effect, not a rebasing artifact.
This is a smooth, near-linear pattern -- much closer to population's
shape than to crime/GDP's irregular swings -- so linear regression is
used deliberately, matching population's justification, not GDP's.

NO REBASING DISCONTINUITY (unlike population's 2020 census jump):
CPI series like this ARE sometimes reset to a new base year (index
snapped back to 100), which would create the same kind of break
population's data had. Checked for it explicitly: the index never
resets anywhere in this series (min 99.3, max 143.8, continuously
rising) -- so unlike population, the FULL 2010-2025 history is used
for the trend fit, not a restricted recent window.

GRANULARITY NOTE: source data is MONTHLY, but this module aggregates
to ANNUAL averages (mean of available months per year) to stay
consistent with every other category's yearly granularity, and
because the rating only ever needs a value for a given visit-date's
YEAR, not a specific month. 2026 is excluded from training (only 7 of
12 months available at time of writing -- an incomplete year's
average isn't comparable to a full year's).

WORKFLOW: this module needs no separate training/saving step like the
RNN/LSTM files -- linear regression is fast enough to fit live, on
demand, same as population_score.py. Run this file directly to
generate the full CSV export and see the affordability direction note
below; nothing needs training or caching for the app to use it.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

from gdp_forecast import NAME_ALIASES  # already extended with Labuan/Putrajaya

DATA_DIR = Path(__file__).parent
SOURCE_CSV = DATA_DIR / 'cpi_2d_state.csv'
FORECAST_CSV = DATA_DIR / 'cpi_forecast.csv'

LAST_REAL_YEAR = 2025  # last COMPLETE year (2026 only has 7 months so far)
FORECAST_YEARS = [2026, 2027, 2028]


def _load_annual_cpi():
    """Annual mean CPI ('overall' division) per state, complete years only."""
    df = pd.read_csv(SOURCE_CSV)
    df = df[df['division'] == 'overall'].copy()
    df['year'] = pd.to_datetime(df['date']).dt.year

    counts = df.groupby(['state', 'year']).size().reset_index(name='n_months')
    complete = counts[counts['n_months'] == 12][['state', 'year']]

    annual = df.groupby(['state', 'year'])['index'].mean().reset_index()
    annual = annual.merge(complete, on=['state', 'year'])  # drop incomplete years
    return annual


_annual_cache = None


def predict_state_cpi(state_name, year):
    """Return predicted annual-average CPI index for a state + year, or
    None if the state isn't found. Uses the full available history
    (see METHODOLOGY NOTE -- no rebasing break to work around here)."""
    global _annual_cache
    if _annual_cache is None:
        _annual_cache = _load_annual_cpi()

    crime_name = NAME_ALIASES.get(state_name, state_name)
    rows = _annual_cache[_annual_cache['state'] == crime_name]
    if rows.empty:
        return None

    model = LinearRegression().fit(rows[['year']], rows['index'])
    return float(model.predict(pd.DataFrame({'year': [year]}))[0])


def get_state_cpi(state_name, year):
    """Blended lookup: real annual average for year <= 2025, linear
    trend forecast for 2026-2028 (clamped beyond that)."""
    global _annual_cache
    if _annual_cache is None:
        _annual_cache = _load_annual_cpi()

    crime_name = NAME_ALIASES.get(state_name, state_name)

    if year <= LAST_REAL_YEAR:
        row = _annual_cache[(_annual_cache['state'] == crime_name) & (_annual_cache['year'] == year)]
        return float(row['index'].iloc[0]) if len(row) else None

    clamped_year = min(FORECAST_YEARS[-1], year)
    return predict_state_cpi(state_name, clamped_year)


def forecast_all_states():
    """Predict every state's CPI for every FORECAST_YEARS year. Returns
    a long-format DataFrame: state, year, predicted_cpi."""
    global _annual_cache
    if _annual_cache is None:
        _annual_cache = _load_annual_cpi()
    states = sorted(_annual_cache['state'].unique())

    rows = []
    for state in states:
        for year in FORECAST_YEARS:
            rows.append({
                'state': state,
                'year': year,
                'predicted_cpi': predict_state_cpi(state, year),
            })
    return pd.DataFrame(rows)


def load_forecast_table():
    if not FORECAST_CSV.exists():
        return None
    return pd.read_csv(FORECAST_CSV)


ALL_STATES = ['Johor', 'Kedah', 'Kelantan', 'Melaka', 'Negeri Sembilan', 'Pahang',
              'Perak', 'Perlis', 'Pulau Pinang', 'Sabah', 'Sarawak', 'Selangor',
              'Terengganu', 'W.P. Kuala Lumpur', 'W.P. Labuan', 'W.P. Putrajaya']


def get_chart_series(state_name):
    """Return {'years': [...], 'values': [...], 'real_count': N} for
    the "Show more detail" chart -- real 2010-2025 (annual averages),
    forecast 2026-2028."""
    years = list(range(2010, FORECAST_YEARS[-1] + 1))
    values = [get_state_cpi(state_name, y) for y in years]
    real_count = sum(1 for y in years if y <= LAST_REAL_YEAR)
    return {'years': years, 'values': values, 'real_count': real_count}


def affordability_score(state_name, year):
    """Return (score 0-100, reason). Relative ranking by CPI across
    all states for the given year -- INVERTED direction from every
    other relative-ranking score in this project: a LOWER cost-of-
    living index means MORE affordable, so it gets the HIGHER score.
    (Flagged loudly here because copying the "higher raw value = higher
    score" pattern from population_score()/gdp_score() unchanged would
    be a real bug, not just a different judgment call.)"""
    values = {}
    for s in ALL_STATES:
        v = get_state_cpi(s, year)
        if v is not None:
            values[s] = v

    crime_name = NAME_ALIASES.get(state_name, state_name)
    if crime_name not in values or len(values) < 2:
        return 75, 'No CPI data available for this state/year; a neutral score is used.'

    series = pd.Series(values)
    lo, hi = series.min(), series.max()
    this_value = series[crime_name]
    # NOTE: (hi - this_value), not (this_value - lo) -- lower CPI wins.
    score = 100 if hi == lo else round(100 * (hi - this_value) / (hi - lo))
    reason = f'Cost-of-living index ~{this_value:.1f} in {year}, relative to other states (lower = more affordable = higher score).'
    return score, reason


if __name__ == '__main__':
    print("Quick spot-check:")
    for state in ['Selangor', 'Sabah', 'Perlis']:
        for yr in [2026, 2027, 2028]:
            print(f"  {state} {yr}: index {predict_state_cpi(state, yr):.1f}")

    print("\nAFFORDABILITY DIRECTION NOTE: a HIGHER index means prices")
    print("are HIGHER (less affordable) -- the opposite direction from")
    print("safety_score(), where a higher score meant safer. Any future")
    print("rating factor built on this should invert the scale (higher")
    print("CPI -> LOWER affordability score), not copy safety_score()'s")
    print("direction unchanged.")

    print("\nForecasting all states...")
    forecast = forecast_all_states()
    forecast['predicted_cpi'] = forecast['predicted_cpi'].round(2)
    forecast.to_csv(FORECAST_CSV, index=False)
    print(f"Saved {len(forecast)} rows to {FORECAST_CSV}")

    print("\nBlended lookup spot-check:")
    for state, year in [('Selangor', 2024), ('Selangor', 2027)]:
        print(f"  {state} {year}: {get_state_cpi(state, year):.2f}")
