"""
population_score.py -- state and district population trend projection,
using linear regression (NOT RNN/LSTM -- see METHODOLOGY NOTE), plus
per-capita normalisation for the crime safety score.

METHODOLOGY NOTE (for the report):
Population growth is smooth and near-linear year to year (checked:
Selangor's 2021-2026 growth sits between 0.3% and 2.3% per year, no
sharp swings) -- unlike crime, which genuinely needed a recurrent model
to capture irregular year-to-year movement. Forcing an LSTM onto a
near-linear trend on ~8 data points would risk overfitting noise rather
than learning anything a straight line doesn't already capture, and
would be a mismatch between model complexity and data complexity.
Linear regression is used deliberately, not as a shortcut.

TRAINING WINDOW: state-level population estimates were rebased against
the 2020 Census, causing a one-off, non-random jump/drop in the 2020
data point for every state (e.g. Selangor +7.5%, Sabah -12.4% in a
single year -- not real demographic swings). Fitting a line across the
full 1970-2026 history would blend two different counting methodologies
and bias predictions for recent/future years specifically (verified:
a full-history fit systematically overpredicts Sabah's population by a
growing margin every year after 2020). This module therefore fits only
on YEAR_FLOOR onward (2021), a single clean window -- not two separate
models. District population data only exists from 2020 onward anyway,
so this issue does not apply there.

DISTRICT POPULATION COVERAGE (for the report): crime_district.csv uses
POLICE district boundaries; population_district.csv uses standard
ADMINISTRATIVE districts. These are different official boundary systems
-- e.g. Johor Bahru is one administrative district but five separate
police districts (Johor Bahru Utara, Johor Bahru Selatan, Iskandar
Puteri, Nusajaya, Seri Alam). Only where the two datasets share an
EXACT district name (115 of 159 crime districts, ~72%) is a safe,
verified per-capita crime rate computed; elsewhere, crime_forecast.py's
existing raw-count ranking is used instead, since fabricating a
population split across mismatched boundaries would be a worse error
than the raw-count limitation it would replace. The safety_score
reason text always states which method was actually used.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression

from crime_forecast import NAME_ALIASES, FORECAST_YEARS  # reuse the same state-name map and horizon

DATA_DIR = Path(__file__).parent
STATE_POP_CSV = DATA_DIR / 'population_state.csv'
DISTRICT_POP_CSV = DATA_DIR / 'population_district.csv'
STATE_FORECAST_CSV = DATA_DIR / 'population_forecast.csv'
DISTRICT_FORECAST_CSV = DATA_DIR / 'population_forecast_district.csv'

YEAR_FLOOR = 2021  # exclude the 2020 census-rebasing discontinuity


def _load_state_pop():
    df = pd.read_csv(STATE_POP_CSV)
    df = df[(df['sex'] == 'both') & (df['age'] == 'overall') & (df['ethnicity'] == 'overall')]
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df[df['year'] >= YEAR_FLOOR]


def _load_district_pop():
    df = pd.read_csv(DISTRICT_POP_CSV)
    df = df[(df['sex'] == 'both') & (df['age'] == 'overall') & (df['ethnicity'] == 'overall')]
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df  # already 2020-2025 only, no rebasing issue to filter out


def _fit_and_predict(years, values, target_year):
    model = LinearRegression().fit(
        pd.DataFrame({'year': years}), values
    )
    return float(model.predict(pd.DataFrame({'year': [target_year]}))[0])


_state_pop_cache = None
_district_pop_cache = None


def predict_state_population(state_name, year):
    """Return predicted population (thousands) for a state + year, or
    None if the state isn't found."""
    global _state_pop_cache
    if _state_pop_cache is None:
        _state_pop_cache = _load_state_pop()

    crime_name = NAME_ALIASES.get(state_name, state_name)
    rows = _state_pop_cache[_state_pop_cache['state'] == crime_name]
    if rows.empty:
        return None
    return _fit_and_predict(rows['year'], rows['population'], year)


def predict_district_population(state_name, district_name, year):
    """Return predicted population (thousands) for a district + year,
    or None if that exact district name isn't in the population data
    (see DISTRICT POPULATION COVERAGE above)."""
    global _district_pop_cache
    if _district_pop_cache is None:
        _district_pop_cache = _load_district_pop()

    crime_name = NAME_ALIASES.get(state_name, state_name)
    rows = _district_pop_cache[
        (_district_pop_cache['state'] == crime_name) &
        (_district_pop_cache['district'] == district_name)
    ]
    if rows.empty or rows['year'].nunique() < 2:
        return None
    return _fit_and_predict(rows['year'], rows['population'], year)


def crime_rate_per_capita(crimes, population_thousands):
    """Crimes per 1,000 people -- the standard way crime is normalised
    for fair comparison across places of different sizes."""
    if not population_thousands or population_thousands <= 0:
        return None
    return crimes / population_thousands


def forecast_all_states():
    """Predict every state's population for every FORECAST_YEARS year
    (2024-2028, same horizon as the crime forecast). Returns a
    long-format DataFrame: state, year, predicted_population."""
    global _state_pop_cache
    if _state_pop_cache is None:
        _state_pop_cache = _load_state_pop()
    states = sorted(_state_pop_cache['state'].unique())

    rows = []
    for state in states:
        for year in FORECAST_YEARS:
            rows.append({
                'state': state,
                'year': year,
                'predicted_population': predict_state_population(state, year),
            })
    return pd.DataFrame(rows)


def forecast_all_districts():
    """Predict every district's population for every FORECAST_YEARS
    year. Only includes districts with at least 2 years of source data
    (predict_district_population() returns None otherwise, and those
    rows are skipped -- this file only ever contains real predictions,
    never silent placeholders)."""
    global _district_pop_cache
    if _district_pop_cache is None:
        _district_pop_cache = _load_district_pop()
    pairs = _district_pop_cache[['state', 'district']].drop_duplicates()

    rows = []
    for _, r in pairs.iterrows():
        for year in FORECAST_YEARS:
            pop = predict_district_population(r['state'], r['district'], year)
            if pop is not None:
                rows.append({
                    'state': r['state'],
                    'district': r['district'],
                    'year': year,
                    'predicted_population': pop,
                })
    return pd.DataFrame(rows)


def load_state_population_forecast():
    """Fast path: read the saved state-level forecast file."""
    if not STATE_FORECAST_CSV.exists():
        return None
    return pd.read_csv(STATE_FORECAST_CSV)


def get_chart_series(state_name):
    """Return {'years': [...], 'values': [...], 'real_count': N} for
    the "Show more detail" chart. Real years are 2021-2026 (the same
    window the trend is fitted on -- see YEAR_FLOOR and the module
    docstring for why pre-2021 data is excluded, the 2020 census
    rebasing). Forecast is 2027-2028."""
    global _state_pop_cache
    if _state_pop_cache is None:
        _state_pop_cache = _load_state_pop()

    crime_name = NAME_ALIASES.get(state_name, state_name)
    real_rows = _state_pop_cache[_state_pop_cache['state'] == crime_name].sort_values('year')
    real_years = list(real_rows['year'])
    real_values = list(real_rows['population'])

    forecast_years = [y for y in [2027, 2028] if y not in real_years]
    forecast_values = [predict_state_population(state_name, y) for y in forecast_years]

    years = real_years + forecast_years
    values = real_values + forecast_values
    return {'years': years, 'values': values, 'real_count': len(real_years)}


def population_score(state_name, year):
    """Return (score 0-100, reason). Relative ranking by population
    across all 16 states for the given year.

    ASSUMPTION (confirm/flip if you disagree): HIGHER population ->
    HIGHER score, on the basis that a bigger state generally has more
    infrastructure/amenities for a visitor. This is a judgment call,
    not a DOSM-derived fact -- reasonable people could argue the
    opposite (a smaller, quieter state scores better for some
    travellers). Swap (this_value - lo) for (hi - this_value) below
    to invert it if you'd rather score the other way."""
    global _state_pop_cache
    if _state_pop_cache is None:
        _state_pop_cache = _load_state_pop()
    states = sorted(_state_pop_cache['state'].unique())

    values = {s: predict_state_population(s, year) for s in states}
    values = {s: v for s, v in values.items() if v is not None}

    crime_name = NAME_ALIASES.get(state_name, state_name)
    if crime_name not in values or len(values) < 2:
        return 75, 'No population data available for this state/year; a neutral score is used.'

    series = pd.Series(values)
    ranked = series.rank(method='min')
    this_rank = int(ranked[crime_name])
    n = len(series)
    score = round(100 * (this_rank - 1) / (n - 1))
    this_value = series[crime_name]
    reason = f'Estimated population ~{this_value:,.0f} thousand in {year}, relative to other states (higher = more infrastructure/amenities, assumption).'
    return score, reason


def load_district_population_forecast():
    """Fast path: read the saved district-level forecast file."""
    if not DISTRICT_FORECAST_CSV.exists():
        return None
    return pd.read_csv(DISTRICT_FORECAST_CSV)


if __name__ == '__main__':

    print("\nForecasting all states...")
    state_forecast = forecast_all_states()
    state_forecast['predicted_population'] = state_forecast['predicted_population'].round(1)
    state_forecast.to_csv(STATE_FORECAST_CSV, index=False)
    print(f"Saved {len(state_forecast)} rows to {STATE_FORECAST_CSV}")

    print("Forecasting all districts (this takes a little longer)...")
    district_forecast = forecast_all_districts()
    district_forecast['predicted_population'] = district_forecast['predicted_population'].round(1)
    district_forecast.to_csv(DISTRICT_FORECAST_CSV, index=False)
    print(f"Saved {len(district_forecast)} rows to {DISTRICT_FORECAST_CSV}")
